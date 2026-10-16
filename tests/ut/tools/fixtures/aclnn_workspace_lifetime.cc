// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

using aclrtStream = void *;
struct aclOpExecutor {};

namespace fixture {
struct Allocation {
  explicit Allocation(uint64_t size) : bytes(new char[size]) {}
  std::unique_ptr<char[]> bytes;
};

std::unordered_map<void *, std::weak_ptr<Allocation>> allocations;
std::function<int()> queued_handler;
uint64_t workspace_size = 0;
aclrtStream expected_stream = reinterpret_cast<void *>(0x1234);

void remember(const std::shared_ptr<Allocation> &allocation) {
  allocations.emplace(allocation->bytes.get(), allocation);
}

int get_workspace_size(uint64_t *size, aclOpExecutor **) {
  *size = workspace_size;
  return 0;
}

int aclnn_fake(void *workspace, uint64_t size, aclOpExecutor *, aclrtStream stream) {
  assert(size == workspace_size);
  assert(stream == expected_stream);
  if (size != 0) {
    const auto allocation = allocations.find(workspace);
    assert(allocation != allocations.end());
    assert(!allocation->second.expired());
  } else {
    assert(workspace == nullptr);
  }
  return 0;
}
}

namespace at {
class Storage {
 public:
  explicit Storage(void *data) : data_(data) {}
  void *data() const { return data_; }

 private:
  void *data_;
};

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::shared_ptr<fixture::Allocation> allocation)
      : allocation_(std::move(allocation)) {}
  Storage storage() const { return Storage(allocation_->bytes.get()); }

 private:
  std::shared_ptr<fixture::Allocation> allocation_;
};

class TensorOptions {
 public:
  explicit TensorOptions(int) {}
  TensorOptions dtype(int) const { return TensorOptions(0); }
};

Tensor empty(std::initializer_list<uint64_t> sizes, TensorOptions) {
  const auto allocation = std::make_shared<fixture::Allocation>(*sizes.begin());
  fixture::remember(allocation);
  return Tensor(allocation);
}

class Scalar {};
class IntArrayRef {};
template <typename T>
class ArrayRef {};
using TensorList = int;
enum ScalarType { kByte };
}

namespace c10 {
template <typename T>
class optional {};
}

constexpr auto kByte = at::kByte;

namespace torch_npu::utils {
int get_npu_device_type() { return 0; }
}

namespace c10_npu {
class CurrentStream {
 public:
  aclrtStream stream(bool) const { return fixture::expected_stream; }
};
CurrentStream getCurrentNPUStream() { return {}; }
}

namespace at_npu::native {
class OpPreparation {
 public:
  static at::Tensor unsafe_empty_workspace(uint64_t size, aclrtStream stream) {
    assert(stream == fixture::expected_stream);
    return at::empty({size}, at::TensorOptions(0));
  }
};

class OpCommand {
 public:
  OpCommand &Name(const char *) { return *this; }
  void SetCustomHandler(std::function<int()> handler) {
    fixture::queued_handler = std::move(handler);
  }
  void Run() {}
};
}

using InitHugeMemThreadLocal = int (*)(void *, bool);
using UnInitHugeMemThreadLocal = int (*)(void *, bool);
using ReleaseHugeMem = int (*)(void *, bool);

void *GetOpApiFuncAddr(const char *name) {
  if (std::strcmp(name, "aclnnFakeGetWorkspaceSize") == 0) {
    return reinterpret_cast<void *>(fixture::get_workspace_size);
  }
  if (std::strcmp(name, "aclnnFake") == 0) {
    return reinterpret_cast<void *>(fixture::aclnn_fake);
  }
  return nullptr;
}

const char *GetOpApiLibName() { return "fixture"; }
const char *aclGetRecentErrMsg() { return "fixture"; }
void ReleaseConvertTypes(const std::tuple<uint64_t *, aclOpExecutor **> &) {}

std::tuple<uint64_t *, aclOpExecutor **> ConvertTypes(int, uint64_t *size,
                                                       aclOpExecutor **executor) {
  return {size, executor};
}

using WorkspaceSizeFunction = int (*)(uint64_t *, aclOpExecutor **);
WorkspaceSizeFunction ConvertToOpApiFunc(
    const std::tuple<uint64_t *, aclOpExecutor **> &, void *address) {
  return reinterpret_cast<WorkspaceSizeFunction>(address);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function function, Tuple tuple, std::index_sequence<I...>) {
  return function(std::get<I>(tuple)...);
}

template <typename Function, typename Tuple>
auto call(Function function, Tuple tuple) {
  return call(function, tuple,
              std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

#define TORCH_CHECK(condition, ...) assert(condition)

#define EXEC_NPU_CMD_MACRO

void submit(uint64_t size) {
  fixture::workspace_size = size;
  EXEC_NPU_CMD(aclnnFake, 0);
}

int main() {
  submit(64);
  fixture::queued_handler();
  submit(0);
  fixture::queued_handler();
}
