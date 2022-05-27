// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/mlir/service/BenchmarkFactory.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <google/protobuf/wrappers.pb.h>

#include <iostream>
#include <memory>
#include <string>

#include "compiler_gym/envs/mlir/service/MlirUtils.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "compiler_gym/util/StrLenConstexpr.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

namespace fs = boost::filesystem;

using grpc::Status;
using grpc::StatusCode;

using BenchmarkProto = compiler_gym::Benchmark;

namespace compiler_gym::mlir_service {

BenchmarkFactory::BenchmarkFactory(const boost::filesystem::path& workingDirectory,
                                   std::optional<std::mt19937_64> rand,
                                   size_t maxLoadedBenchmarksCount)
    : workingDirectory_(workingDirectory),
      rand_(rand.has_value() ? *rand : std::mt19937_64(std::random_device()())),
      maxLoadedBenchmarksCount_(maxLoadedBenchmarksCount) {
  CHECK(maxLoadedBenchmarksCount) << "Assertion maxLoadedBenchmarksCount > 0 failed! "
                                  << "maxLoadedBenchmarksCount = " << maxLoadedBenchmarksCount;
  VLOG(2) << "BenchmarkFactory initialized";
}

Status BenchmarkFactory::getBenchmark(const BenchmarkProto& benchmarkMessage,
                                      std::unique_ptr<Benchmark>* benchmark) {
  // Check if the benchmark has already been loaded into memory.
  auto loaded = benchmarks_.find(benchmarkMessage.uri());
  if (loaded != benchmarks_.end()) {
    VLOG(3) << "MLIR benchmark cache hit: " << benchmarkMessage.uri();
    *benchmark = loaded->second.clone(workingDirectory_);
    return Status::OK;
  }

  VLOG(3) << "MLIR benchmark cache miss for: " << benchmarkMessage.uri();
  // Benchmark not cached, cache it and try again.
  std::map<std::string, Bitcode> files;
  for (auto& [filename, file] : benchmarkMessage.files()) {
    Bitcode bitcode;
    RETURN_IF_ERROR(processFile(filename, file, &bitcode));
    files[filename] = bitcode;
  }
  // We don't want to make modules out of all the files, so the ones we do are indexed and
  // pointed to by the features
  google::protobuf::Int32Value benchmarkTasksSize;
  benchmarkMessage.features().at("num_tasks").UnpackTo(&benchmarkTasksSize);
  std::vector<std::string> mlirFilenames;
  for (int i = 0; i < benchmarkTasksSize.value(); i++) {
    google::protobuf::StringValue filenameValue;
    benchmarkMessage.features().at(std::to_string(i)).UnpackTo(&filenameValue);
    mlirFilenames.push_back(filenameValue.value());
  }

  RETURN_IF_ERROR(addBenchmark(benchmarkMessage.uri(), files, mlirFilenames,
                               benchmarkMessage.dynamic_config()));

  return getBenchmark(benchmarkMessage, benchmark);
}

Status BenchmarkFactory::addBenchmark(const std::string& uri,
                                      const std::map<std::string, Bitcode>& files,
                                      const std::vector<std::string>& mlirFilenames,
                                      std::optional<BenchmarkDynamicConfig> dynamicConfig) {
  std::map<std::string, std::unique_ptr<mlir::MLIRContext>> contexts;
  std::map<std::string, std::unique_ptr<mlir::OwningOpRef<mlir::ModuleOp>>> modules;

  for (const std::string& filename : mlirFilenames) {
    const std::string& contents = files.at(filename);
    contexts[filename] = mlir::createMlirContext();
    Status status;
    modules[filename] = makeModule(*contexts[filename], contents, uri, &status);
    RETURN_IF_ERROR(status);
  }

  if (benchmarks_.size() == maxLoadedBenchmarksCount_) {
    VLOG(2) << "MLIR benchmark cache reached maximum size " << maxLoadedBenchmarksCount_
            << ". Evicting random 50%.";
    for (int i = 0; i < static_cast<int>(maxLoadedBenchmarksCount_ / 2); ++i) {
      // Select a cached benchmark randomly.
      std::uniform_int_distribution<size_t> distribution(0, benchmarks_.size() - 1);
      size_t index = distribution(rand_);
      auto iterator = std::next(std::begin(benchmarks_), index);

      // Evict the benchmark from the pool of loaded benchmarks.
      benchmarks_.erase(iterator);
    }
  }

  benchmarks_.insert(
      {uri, Benchmark(uri, files, std::move(contexts), std::move(modules),
                      (dynamicConfig.has_value() ? *dynamicConfig : BenchmarkDynamicConfig()),
                      workingDirectory_)});

  VLOG(2) << "Cached MLIR benchmark: " << uri << ". Cache size = " << benchmarks_.size()
          << " items";
  return Status::OK;
}

Status BenchmarkFactory::processFile(const std::string filename, const File& file,
                                     Bitcode* bitcode) {
  switch (file.data_case()) {
    case compiler_gym::File::DataCase::kContents: {
      VLOG(3) << "MLIR benchmark cache miss, add bitcode: " << filename;
      *bitcode = file.contents();
      return Status::OK;
    }
    case compiler_gym::File::DataCase::kUri: {
      VLOG(3) << "MLIR benchmark cache miss, read from URI: " << file.uri();
      // Check the protocol of the benchmark URI.
      if (file.uri().find("file:///") != 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      fmt::format("Invalid benchmark data URI. "
                                  "Only the file:/// protocol is supported: \"{}\"",
                                  file.uri()));
      }

      const fs::path path(file.uri().substr(util::strLen("file:///"), std::string::npos));
      return readBitcodeFile(path, bitcode);
    }
    case compiler_gym::File::DataCase::DATA_NOT_SET:
      return Status(StatusCode::INVALID_ARGUMENT,
                    fmt::format("No program set in file:\n{}", filename));
  }
}

}  // namespace compiler_gym::mlir_service
