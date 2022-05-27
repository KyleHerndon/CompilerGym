# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional

from google.protobuf.any_pb2 import Any
from google.protobuf.wrappers_pb2 import Int32Value, StringValue

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command, File
from compiler_gym.util.runfiles_path import runfiles_path
from compiler_gym.util.shell_format import plural

logger = logging.getLogger(__name__)

_matmul_bin = runfiles_path("compiler_gym/third_party/matmul/matmul/bin/matmul")
_matmul_includes = runfiles_path(
    "compiler_gym/third_party/matmul/matmul/include/matmul-2.3.0"
)

ensembles = {
    "ensemble_simple_sizes": [
        (64, 64, 64),
        (64, 64, 128),
        (64, 128, 64),
        (64, 128, 128),
        (128, 128, 128),
    ],

    "ensemble_small_sizes": [
        (18, 32, 96),
        (24, 64, 96),
        (24, 64, 256),
        (48, 64, 128),
        (192, 64, 128),
        (192, 128, 128),
        (480, 512, 16),
        (384, 256, 256),
    ],

    "ensemble_best_sizes": [),
    	(512, 1024, 1024),
    	(1024, 1024, 512),
    	(512, 1024, 3072),
    	(512, 3072, 1024),
    	(1024, 3072, 512),
    	(3072, 1024, 512),
    	(1, 1024, 1024),
    	(1, 1024, 5),
    	(1024, 5, 1),
    	(1024, 1024, 1),
    	(16, 1024, 512),
    	(1, 5, 1024),
    ],

    "ensemble_resnet_sizes": [
    	(12544, 64, 147),
    	(3136, 64, 64),
    	(3136, 64, 576),
    	(3136, 256, 64),
    	(3136, 64, 256),
    	(3136, 128, 256),
    	(784, 128, 1152),
    	(784, 512, 128),
    	(784, 512, 256),
    	(784, 128, 512),
    	(784, 256, 512),
    	(196, 256, 2304),
    	(196, 1024, 256),
    	(196, 1024, 512),
    	(196, 256, 1024),
    	(196, 512, 1024),
    	(49, 512, 4608),
    	(49, 2048, 512),
    	(49, 2048, 1024),
    	(49, 512, 2048),
    ]
}

def new_any(msg):
    any = Any()
    any.Pack(msg)
    return any


def mlir_filename(mnk):
    return "_".join(mnk) + ".mlir"


cc_base_format_string = """
#include <benchmark/benchmark.h>
#include <mlir/ExecutionEngine/RunnerUtils.h>

#include <cstdio>
#include <vector>

void naive_matmul(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {{
  // correctness check
  for (size_t i = 0; i < m; i++) {{
    for (size_t j = 0; j < n; j++) {{
#ifdef COLUMN_MAJOR
      size_t ci = i + j * m;
#else
      size_t ci = i * n + j;
#endif
      c[ci] = 0.0f;
      for (size_t p = 0; p < k; p++) {{
#ifdef COLUMN_MAJOR
        c[ci] += a[i + p * m] * b[p + j * k];
#else
        c[ci] += a[i * k + p] * b[p * n + j];
#endif
      }}
    }}
  }}
}}

void init_matrix(float* a, int nrows, int ncols) {{
  for (int j = 0; j < ncols; j++) {{
    for (int i = 0; i < nrows; i++) {{
      a[i + j * nrows] = ((float)rand() / (float)RAND_MAX);
    }}
  }}
}}

size_t g_errors = 0;

{BENCHMARKS}

int main(int argc, char** argv) {{
  benchmark::Initialize(&argc, argv);
  {REGISTER_BENCHMARKS}
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return g_errors != 0;
}}
"""

cc_benchmark_format = """
extern "C" {{
void matmul_{M}_{N}_{K}(float* aligned_a, float* allocated_a, int64_t offset_a, int64_t size_a0,
            int64_t size_a1, int64_t strides_a0, int64_t strides_a1, float* aligned_b,
            float* allocated_b, int64_t offset_b, int64_t size_b0, int64_t size_b1,
            int64_t strides_b0, int64_t strides_b1, float* aligned_c, float* allocated_c,
            int64_t offset_c, int64_t size_c0, int64_t size_c1, int64_t strides_c0,
            int64_t strides_c1);
}}

static void BenchmarkFunction_{M}_{N}_{K}(benchmark::State& state) {{
  int MDIM = {M};
  int NDIM = {N};
  int KDIM = {K};
  std::vector<float> a(MDIM * KDIM);
  std::vector<float> b(KDIM * NDIM);
  std::vector<float> c(MDIM * NDIM);
  float *A = a.data(), *B = b.data(), *C = c.data();
  init_matrix(A, MDIM, KDIM);
  init_matrix(B, KDIM, NDIM);
  init_matrix(C, MDIM, NDIM);
  int LDA = KDIM;
  int LDB = NDIM;
  int LDC = NDIM;

  for (auto _ : state) {{
    matmul_{M}_{N}_{K}(A, A, 0, MDIM, KDIM, LDA, 1, B, B, 0, KDIM, NDIM, LDB, 1, C, C, 0, MDIM, NDIM, LDC, 1);
  }}

  std::vector<float> c2(MDIM * NDIM);
  float* C2 = c2.data();
  size_t errors = 0;
  naive_matmul(A, B, C2, MDIM, KDIM, NDIM);
  for (size_t i = 0; i < MDIM; i++) {{
    for (size_t j = 0; j < NDIM; j++) {{
      size_t ci = i + j * MDIM;
      if (std::abs(C[ci] - C2[ci]) > 0.01f) {{
        if (errors == 0) {{
          fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\\n", i, j, C[ci],
                  C2[ci]);
        }}
        errors++;
      }}
    }}
  }}
  fprintf(stderr, "Detected %ld errors.\\n", errors);
  g_errors = errors;
}}
"""

cc_benchmark_register_format = """
benchmark::RegisterBenchmark("BM_Matmul_{M}_{N}_{K}", BenchmarkFunction_{M}_{N}_{K})
      ->MeasureProcessCPUTime()
      ->UseRealTime();
"""

make_base_format_string = """# Automatically generated makefile for benchmark
ifndef LLVM_INSTALL
$(error LLVM_INSTALL not defined in MLIR benchmark make file)
endif
ifndef BENCHMARK_INSTALL
$(error BENCHMARK_INSTALL not defined in MLIR benchmark make file)
endif
CC=gcc
CFLAGS=-I. -I $(LLVM_INSTALL)/include/ -I $(BENCHMARK_INSTALL)/include/ -lstdc++ -lm -pthread -v -g3
MLIRFLAGS = -Ofast -mllvm -enable-matrix -mllvm -matrix-allow-contract \
\t-mllvm -matrix-default-layout=row-major -g3

build: benchmark_main

benchmark_main: benchmark_main.cc{targets_str}
\t$(CC) -o $@ $^ $(BENCHMARK_INSTALL)/lib/libbenchmark.a $(CFLAGS)

run: build
\t./benchmark_main --benchmark_format=json"""

make_target_format_string = """
{mlir_filename}.o: {mlir_filename}.ll
\t$(CC) -c -o $@ $< $(MLIRFLAGS)"""


def make_and_main(sizes):
    targets_str = ""
    run_str = ""
    make = ""
    cc_benchmarks = ""
    cc_register_benchmarks = ""
    for size in sizes:
        mlir_filename = mlir_filename(size)
        (m, n, k) = size
        cc_benchmarks += cc_benchmark_format.format(M=m, N=n, K=k)
        cc_register_benchmarks += cc_benchmark_register_format.format(M=m, N=n, K=k)
        obj_filename = mlir_filename + ".o"
        targets_str += " " + obj_filename
        make += make_target_format_string.format(mlir_filename=mlir_filename)
    make = (
        make_base_format_string.format(targets_str=targets_str, run_str=run_str) + make
    )
    cc = cc_base_format_string.format(
        BENCHMARKS=cc_benchmarks, REGISTER_BENCHMARKS=cc_register_benchmarks
    )
    return make, cc


class MatmulBenchmark(Benchmark):
    """A matmul benchmark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._src = None
        self.proto.dynamic_config.MergeFrom(
            BenchmarkDynamicConfig(
                build_cmd=Command(
                    argument=["$CC", "$IN"],
                    outfile=["benchmark_main"],
                    timeout_seconds=600,
                ),
                run_cmd=Command(
                    argument=["./benchmark_main", "--benchmark_format=json"],
                    timeout_seconds=300,
                ),
            )
        )


class MatmulDataset(Dataset):
    """A dataset which generates matmul programs."""

    def __init__(
        self,
        site_data_base: Path,
        sort_order: int = 0,
        matmul_bin: Optional[Path] = None,
        matmul_includes: Optional[Path] = None,
    ):
        """Constructor.
        :param site_data_base: The base path of a directory that will be used to
            store installed files.
        :param sort_order: An optional numeric value that should be used to
            order this dataset relative to others. Lowest value sorts first.
        :param matmul_bin: The path of the matmul binary to use. If not
            provided, the version of matmul shipped with CompilerGym is used.
        :param matmul_includes: The path of the matmul includes directory. If
            not provided, the includes of the matmul shipped with CompilerGym is
            used.
        """
        super().__init__(
            name="generator://matmul-v0",
            description="Targeted size matmul programs",
            references={},
            license="MIT",
            site_data_base=site_data_base,
            sort_order=sort_order,
            benchmark_class=MatmulBenchmark,
        )
        self.matmul_bin_path = matmul_bin or _matmul_bin
        self.matmul_includes_path = matmul_includes or _matmul_includes

    @property
    def size(self) -> int:
        return len(matmul_sizes)

    def name_from_size(self, mnk):
        return f"{self.name}/{mnk[0]}_{mnk[1]}_{mnk[2]}"

    # TODO(kyleherndon): Benchmarks are actually dynamically generated for any
    # provided parameters, figure out a better way to represent this in the list of
    # available benchmarks
    def benchmark_uris(self) -> Iterable[str]:
        return (self.name_from_size(mnk) for mnk in matmul_sizes)

    def benchmark(self, uri: str) -> MatmulBenchmark:
        sizestr = uri.split("/")[-1]
        features = dict()
        files = dict()

        if sizestr not in ensembles:
            # Perform one size of matmul during the benchmark
            sizes = [(int(i) for i in sizestr.split("_"))]
        else:
            # Look up what sizes of matmul to perform for the specified ensemble
            sizes = ensembles[sizestr]
        features["num_tasks"] = new_any(Int32Value(value=len(sizes)))
        for benchmark_size in sizes:
            files.update(self.benchmark_files_from_size(benchmark_size))
        for index, (filename, _) in enumerate(files.items()):
            features[str(index)] = new_any(StringValue(value=filename))
        make_content, main_content = make_and_main(features)
        files["Makefile"] = File(contents=make_content.encode())
        files["benchmark_main.cc"] = File(contents=main_content.encode())
        return MatmulBenchmark.from_sources(uri, files, features)


    def benchmark_files_from_size(
        self, mnk, max_retries: int = 3, retry_count: int = 0
    ) -> Mapping[str, File]:
        """Get a benchmark from a uint32 seed.
        :param mnk: 3-tuple containing m, n, k sizes of the matmul
        :return: A benchmark instance.
        :raises OSError: If matmul fails.
        :raises BenchmarkInitError: If the C program generated by matmul cannot
            be lowered to mlir-IR.
        """
        if retry_count >= max_retries:
            raise OSError(
                f"matmul failed after {retry_count} {plural(retry_count, 'attempt', 'attempts')} "
                f"with size {mnk}"
            )

        self.install()
        mnk = list(mnk)
        # Run matmul with the given size and regex to produce the correct mlir
        logger.debug("Exec matmul --mnk %d", mnk)

        # TODO(kyleherndon): refactor these to another location
        src_content = """
func @matmul_${M}_${N}_${K}(%a: tensor<${M}x${K}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
             %b: tensor<${K}x${N}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
             %c: tensor<${M}x${N}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>}) -> tensor<${M}x${N}xf32>
attributes { passthrough = [["target-cpu", "haswell"], ["prefer-vector-width", "256"]]}
{
  %f0 = arith.constant 0.0 : f32
  %f1 = linalg.fill(%f0, %c) : f32, tensor<${M}x${N}xf32> -> tensor<${M}x${N}xf32>
  %d = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>)
    outs(%f1: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  return %d : tensor<${M}x${N}xf32>
}"""
        filename = "matmul"
        for i in mnk:
            filename += "_" + str(i)
        mlir_filename = filename + ".mlir"
        new_content = src_content.replace("${M}", str(mnk[0]))
        new_content = new_content.replace("${N}", str(mnk[1]))
        content = new_content.replace("${K}", str(mnk[2]))
        files = dict()
        files[mlir_filename] = File(contents=content.encode())
        return files
