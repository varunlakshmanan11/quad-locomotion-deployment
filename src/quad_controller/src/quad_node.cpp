#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <chrono>
#include <fstream>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "quad_controller/observation_builder.hpp"
#include "quad_controller/policy_engine.hpp"

#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ROS 2 glue.
//
// This node does no policy logic of its own: it converts messages into a
// RobotState, hands that to the observation builder, runs the engine, and
// publishes the resulting joint targets. Everything worth testing lives in the
// library rather than here.
// ---------------------------------------------------------------------------

namespace {
constexpr int CONTROL_HZ = 50;
}

class QuadNode : public rclcpp::Node {
public:
    QuadNode() : Node("quad_node") {
        declare_parameter("cmd_vx", 0.5);
        declare_parameter("cmd_vy", 0.0);
        declare_parameter("cmd_wz", 0.0);
        command_.vx = static_cast<float>(get_parameter("cmd_vx").as_double());
        command_.vy = static_cast<float>(get_parameter("cmd_vy").as_double());
        command_.wz = static_cast<float>(get_parameter("cmd_wz").as_double());

        const std::string engine_path =
            ament_index_cpp::get_package_share_directory("quad_controller") +
            "/models/policy.engine";

        try {
            engine_ = std::make_unique<quad::TensorRtPolicy>(engine_path);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "failed to load policy: %s", e.what());
            return;
        }
        RCLCPP_INFO(get_logger(), "loaded %s", engine_->describe().c_str());

        joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&QuadNode::onJointState, this, std::placeholders::_1));

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10,
            std::bind(&QuadNode::onOdom, this, std::placeholders::_1));

        cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10,
            std::bind(&QuadNode::onCmd, this, std::placeholders::_1));

        scan_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
            "height_scan", 10,
            std::bind(&QuadNode::onHeightScan, this, std::placeholders::_1));

        cmd_pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_command", 10);

        // Control now runs from the joint-state callback; see onJointState.

        RCLCPP_INFO(get_logger(), "quad_node ready, control loop at %d Hz", CONTROL_HZ);
    }

private:
    // Resolve joint order by name once, rather than trusting message ordering.
    void buildIndexMap(const std::vector<std::string>& names) {
        index_map_.assign(quad::ACT_DIM, -1);
        for (int i = 0; i < quad::ACT_DIM; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            for (std::size_t j = 0; j < names.size(); ++j) {
                if (names[j] == quad::JOINT_NAMES[idx]) {
                    index_map_[idx] = static_cast<int>(j);
                    break;
                }
            }
            if (index_map_[idx] < 0) {
                RCLCPP_ERROR(get_logger(), "joint %s missing from /joint_states",
                             quad::JOINT_NAMES[idx]);
            }
        }
    }

    void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (index_map_.empty()) buildIndexMap(msg->name);

        for (int i = 0; i < quad::ACT_DIM; ++i) {
            const auto dst = static_cast<std::size_t>(i);
            const int  src = index_map_[dst];
            if (src < 0) return;
            const auto s = static_cast<std::size_t>(src);
            if (s < msg->position.size())
                state_.joint_positions[dst] = static_cast<float>(msg->position[s]);
            if (s < msg->velocity.size())
                state_.joint_velocities[dst] = static_cast<float>(msg->velocity[s]);
        }
        state_stamp_ = Clock::now();
        have_joints_ = true;

        // Event-driven: act on state the moment it arrives rather than waiting
        // for a timer tick, which costs half a control period on average.
        control();
    }

    void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
        state_.base_linear_velocity = {
            static_cast<float>(msg->twist.twist.linear.x),
            static_cast<float>(msg->twist.twist.linear.y),
            static_cast<float>(msg->twist.twist.linear.z)};

        // IsaacComputeOdometry reports angular velocity in degrees per second;
        // the policy was trained on radians.
        constexpr float kDegToRad = 0.0174532925f;
        state_.base_angular_velocity = {
            static_cast<float>(msg->twist.twist.angular.x) * kDegToRad,
            static_cast<float>(msg->twist.twist.angular.y) * kDegToRad,
            static_cast<float>(msg->twist.twist.angular.z) * kDegToRad};

        state_.projected_gravity = quad::projectedGravity(
            static_cast<float>(msg->pose.pose.orientation.x),
            static_cast<float>(msg->pose.pose.orientation.y),
            static_cast<float>(msg->pose.pose.orientation.z),
            static_cast<float>(msg->pose.pose.orientation.w));

        have_odom_ = true;
    }

    void onCmd(const geometry_msgs::msg::Twist::SharedPtr msg) {
        command_.vx = static_cast<float>(msg->linear.x);
        command_.vy = static_cast<float>(msg->linear.y);
        command_.wz = static_cast<float>(msg->angular.z);
    }

    void onHeightScan(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        if (msg->data.size() != quad::SCAN_DIM) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "height scan has %zu values, expected %d",
                                 msg->data.size(), quad::SCAN_DIM);
            return;
        }
        std::copy(msg->data.begin(), msg->data.end(), state_.height_scan.begin());
        have_scan_ = true;
    }

    void control() {
        if (!engine_ || !have_joints_ || !have_odom_ || !have_scan_) return;

        const quad::Observation obs = builder_.build(state_, command_);

        const auto t_before = Clock::now();

        quad::JointArray action{};
        try {
            action = engine_->infer(obs);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "inference failed: %s", e.what());
            return;
        }

        const auto t_after = Clock::now();
        builder_.setPreviousAction(action);

        const quad::JointArray targets = quad::actionToJointTargets(action);

        sensor_msgs::msg::JointState out;
        out.header.stamp = now();
        out.name.assign(quad::JOINT_NAMES.begin(), quad::JOINT_NAMES.end());
        out.position.assign(targets.begin(), targets.end());
        cmd_pub_->publish(out);

        record(t_before, t_after, Clock::now());
    }

    // ---- latency instrumentation ------------------------------------------
    // Inference time alone understates what the control loop actually costs.
    // These also capture the wait between a state message arriving and the
    // matching command going out, which is what the robot experiences.
    using Clock = std::chrono::steady_clock;

    struct Sample {
        double queue_us;      // state arrival -> start of inference
        double inference_us;  // engine forward pass
        double total_us;      // state arrival -> command published
    };

    static double microseconds(Clock::time_point from, Clock::time_point to) {
        return std::chrono::duration<double, std::micro>(to - from).count();
    }

    void record(Clock::time_point before, Clock::time_point after,
                Clock::time_point published) {
        samples_.push_back({microseconds(state_stamp_, before),
                            microseconds(before, after),
                            microseconds(state_stamp_, published)});

        if (samples_.size() >= kFlushEvery) {
            flush();
        }
    }

    void flush() {
        const bool fresh = !header_written_;
        std::ofstream csv(kLatencyPath, fresh ? std::ios::trunc : std::ios::app);
        if (!csv) return;

        if (fresh) {
            csv << "queue_us,inference_us,total_us\n";
            header_written_ = true;
        }
        for (const Sample& s : samples_) {
            csv << s.queue_us << ',' << s.inference_us << ',' << s.total_us << '\n';
        }

        double inf_sum = 0.0;
        double tot_sum = 0.0;
        double tot_max = 0.0;
        for (const Sample& s : samples_) {
            inf_sum += s.inference_us;
            tot_sum += s.total_us;
            tot_max = std::max(tot_max, s.total_us);
        }
        const auto n = static_cast<double>(samples_.size());
        RCLCPP_INFO(get_logger(),
                    "latency over %zu ticks: inference %.1f us, end-to-end %.1f us "
                    "(max %.1f us)",
                    samples_.size(), inf_sum / n, tot_sum / n, tot_max);

        samples_.clear();
    }

    static constexpr const char* kLatencyPath = "/tmp/quad_latency.csv";
    static constexpr std::size_t kFlushEvery = 500;

    Clock::time_point    state_stamp_{};
    std::vector<Sample>  samples_;
    bool                 header_written_{false};

    std::unique_ptr<quad::IPolicyEngine> engine_;
    quad::ObservationBuilder             builder_;
    quad::RobotState                     state_;
    quad::VelocityCommand                command_;

    std::vector<int> index_map_;
    bool have_joints_{false};
    bool have_odom_{false};
    bool have_scan_{false};

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr    cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr scan_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr    cmd_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<QuadNode>());
    rclcpp::shutdown();
    return 0;
}
