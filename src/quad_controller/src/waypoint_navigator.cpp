#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Waypoint navigator.
//
// Sits above the locomotion policy: reads the robot pose from /odom, steers
// toward the active waypoint by publishing a velocity command on /cmd_vel,
// and advances once the waypoint is reached.
//
//   /odom  ->  [ this node ]  ->  /cmd_vel  ->  [ policy node ]  ->  joints
//
// Control law is a proportional controller on heading and distance, with a
// turn-in-place phase when the heading error is large. A flat-terrain gait is
// far more stable walking forward than sideways, so the robot points itself at
// the target before translating.
// ---------------------------------------------------------------------------

namespace {

struct Waypoint {
    double x;
    double y;
};

// Wrap an angle to (-pi, pi].
double wrapAngle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

}  // namespace

class WaypointNavigator : public rclcpp::Node {
public:
    WaypointNavigator() : Node("waypoint_navigator") {
        // Course is configurable; the default is a square.
        declare_parameter("waypoints_x", std::vector<double>{ 2.0,  2.0, 0.0, 0.0});
        declare_parameter("waypoints_y", std::vector<double>{ 0.0,  2.0, 2.0, 0.0});
        declare_parameter("arrival_tolerance", 0.35);
        declare_parameter("heading_tolerance", 0.30);
        declare_parameter("max_linear_speed", 0.8);
        declare_parameter("max_angular_speed", 0.8);
        declare_parameter("loop_course", true);

        const auto xs = get_parameter("waypoints_x").as_double_array();
        const auto ys = get_parameter("waypoints_y").as_double_array();
        if (xs.size() != ys.size() || xs.empty()) {
            RCLCPP_ERROR(get_logger(), "waypoints_x and waypoints_y must be equal length and non-empty");
            return;
        }
        for (size_t i = 0; i < xs.size(); ++i) waypoints_.push_back({xs[i], ys[i]});

        arrival_tol_  = get_parameter("arrival_tolerance").as_double();
        heading_tol_  = get_parameter("heading_tolerance").as_double();
        max_linear_   = get_parameter("max_linear_speed").as_double();
        max_angular_  = get_parameter("max_angular_speed").as_double();
        loop_course_  = get_parameter("loop_course").as_bool();

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10,
            std::bind(&WaypointNavigator::onOdom, this, std::placeholders::_1));

        cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

        timer_ = create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&WaypointNavigator::control, this));

        RCLCPP_INFO(get_logger(), "waypoint_navigator ready, %zu waypoints", waypoints_.size());
    }

private:
    void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
        x_ = msg->pose.pose.position.x;
        y_ = msg->pose.pose.position.y;

        // yaw from the quaternion
        const double qx = msg->pose.pose.orientation.x;
        const double qy = msg->pose.pose.orientation.y;
        const double qz = msg->pose.pose.orientation.z;
        const double qw = msg->pose.pose.orientation.w;
        yaw_ = std::atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz));

        have_pose_ = true;
    }

    void control() {
        if (!have_pose_ || waypoints_.empty()) return;

        if (finished_) {
            publish(0.0, 0.0, 0.0);
            return;
        }

        const Waypoint& target = waypoints_[index_];
        const double dx = target.x - x_;
        const double dy = target.y - y_;
        const double distance = std::hypot(dx, dy);

        if (distance < arrival_tol_) {
            RCLCPP_INFO(get_logger(), "reached waypoint %zu (%.2f, %.2f)",
                        index_, target.x, target.y);
            advance();
            return;
        }

        const double desired_yaw = std::atan2(dy, dx);
        const double yaw_error   = wrapAngle(desired_yaw - yaw_);

        double vx = 0.0;
        double wz = std::clamp(1.5 * yaw_error, -max_angular_, max_angular_);

        // Only translate once roughly pointing at the target.
        if (std::fabs(yaw_error) < heading_tol_) {
            vx = std::clamp(0.8 * distance, 0.15, max_linear_);
        }

        publish(vx, 0.0, wz);
    }

    void advance() {
        ++index_;
        if (index_ >= waypoints_.size()) {
            if (loop_course_) {
                index_ = 0;
                RCLCPP_INFO(get_logger(), "course complete, looping");
            } else {
                finished_ = true;
                RCLCPP_INFO(get_logger(), "course complete, holding");
            }
        }
    }

    void publish(double vx, double vy, double wz) {
        geometry_msgs::msg::Twist msg;
        msg.linear.x  = vx;
        msg.linear.y  = vy;
        msg.angular.z = wz;
        cmd_pub_->publish(msg);
    }

    std::vector<Waypoint> waypoints_;
    size_t index_{0};
    bool   finished_{false};
    bool   loop_course_{true};

    double arrival_tol_{0.35};
    double heading_tol_{0.30};
    double max_linear_{0.8};
    double max_angular_{0.8};

    double x_{0.0}, y_{0.0}, yaw_{0.0};
    bool   have_pose_{false};

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr  cmd_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WaypointNavigator>());
    rclcpp::shutdown();
    return 0;
}
