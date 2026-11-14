// Convert MaSTrack's watchdog-protected VMOT datagrams to official As2 SportClient calls.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <unitree/robot/a2/sport/sport_client.hpp>

namespace {

constexpr std::uint8_t kIdleSource = 0;
constexpr std::uint8_t kTargetFollowSource = 2;
constexpr float kMaxForwardMps = 0.8F;
constexpr float kMaxLateralMps = 0.5F;
constexpr float kMaxYawRps = 1.0F;

#pragma pack(push, 1)
struct VelocityIntent {
  char magic[4];
  std::uint8_t version;
  std::uint8_t source;
  std::uint16_t intent_id;
  std::uint32_t sequence;
  std::uint64_t steady_ns;
  std::uint32_t ttl_ms;
  float vx;
  float vy;
  float wz;
};
#pragma pack(pop)

static_assert(sizeof(VelocityIntent) == 36, "VMOT wire layout changed");

std::atomic_bool running{true};

void handle_signal(int) { running.store(false); }

std::uint64_t steady_now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

float clamp(float value, float limit) {
  return std::clamp(value, -limit, limit);
}

bool valid_intent(const VelocityIntent &intent, std::uint64_t now_ns) {
  if (std::memcmp(intent.magic, "VMOT", 4) != 0 || intent.version != 1) {
    return false;
  }
  if (intent.source != kIdleSource && intent.source != kTargetFollowSource) {
    return false;
  }
  if (intent.ttl_ms < 20 || intent.ttl_ms > 5000 || intent.steady_ns > now_ns) {
    return false;
  }
  const auto age_ns = now_ns - intent.steady_ns;
  if (age_ns > static_cast<std::uint64_t>(intent.ttl_ms) * 1'000'000ULL) {
    return false;
  }
  if (!std::isfinite(intent.vx) || !std::isfinite(intent.vy) ||
      !std::isfinite(intent.wz)) {
    return false;
  }
  if (intent.source == kIdleSource &&
      (std::abs(intent.vx) > 1e-6F || std::abs(intent.vy) > 1e-6F ||
       std::abs(intent.wz) > 1e-6F)) {
    return false;
  }
  return true;
}

int create_socket(const std::string &path) {
  if (path.size() >= sizeof(sockaddr_un::sun_path)) {
    throw std::runtime_error("Unix socket path is too long");
  }
  struct stat status {};
  if (::lstat(path.c_str(), &status) == 0) {
    if (!S_ISSOCK(status.st_mode)) {
      throw std::runtime_error("refusing to replace non-socket path: " + path);
    }
    if (::unlink(path.c_str()) != 0) {
      throw std::runtime_error("failed to remove stale socket: " + path);
    }
  }
  const int fd = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    throw std::runtime_error("failed to create Unix datagram socket");
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
  if (::bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
    ::close(fd);
    throw std::runtime_error("failed to bind Unix datagram socket: " + path);
  }
  return fd;
}

class SocketCleanup {
 public:
  SocketCleanup(int fd, std::string path) : fd_(fd), path_(std::move(path)) {}
  ~SocketCleanup() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    ::unlink(path_.c_str());
  }

 private:
  int fd_;
  std::string path_;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0]
              << " NETWORK_INTERFACE [VMOT_SOCKET_PATH]\n";
    return 2;
  }
  const std::string network_interface = argv[1];
  const std::string socket_path =
      argc == 3 ? argv[2] : "/tmp/mastrack_motion_intent.sock";

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);
    unitree::robot::a2::SportClient client;
    client.SetTimeout(1.0F);
    client.Init();
    const int recovery_status = client.SetAutoRecovery(1);
    const int fd = create_socket(socket_path);
    SocketCleanup cleanup(fd, socket_path);
    std::cout << "AS2_SPORT_BRIDGE_READY interface=" << network_interface
              << " socket=" << socket_path
              << " auto_recovery_status=" << recovery_status << std::endl;

    bool stopped = false;
    while (running.load()) {
      pollfd descriptor{fd, POLLIN, 0};
      const int poll_status = ::poll(&descriptor, 1, 50);
      if (poll_status <= 0 || !(descriptor.revents & POLLIN)) {
        if (!stopped) {
          client.StopMove();
          stopped = true;
        }
        continue;
      }

      VelocityIntent intent{};
      const auto received = ::recv(fd, &intent, sizeof(intent), 0);
      if (received != static_cast<ssize_t>(sizeof(intent)) ||
          !valid_intent(intent, steady_now_ns())) {
        if (!stopped) {
          client.StopMove();
          stopped = true;
        }
        continue;
      }

      int status = 0;
      if (intent.source == kIdleSource) {
        status = stopped ? 0 : client.StopMove();
        stopped = true;
      } else {
        status = client.Move(clamp(intent.vx, kMaxForwardMps),
                             clamp(intent.vy, kMaxLateralMps),
                             clamp(intent.wz, kMaxYawRps));
        stopped = false;
      }
      if (status != 0) {
        std::cerr << "SportClient command failed status=" << status
                  << " sequence=" << intent.sequence << std::endl;
      }
    }
    client.StopMove();
    std::cout << "AS2_SPORT_BRIDGE_STOPPED" << std::endl;
  } catch (const std::exception &error) {
    std::cerr << "AS2_SPORT_BRIDGE_ERROR " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
