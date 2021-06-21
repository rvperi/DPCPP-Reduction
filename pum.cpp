//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include "align.hpp"

// Summation of 512M 'one' values
constexpr size_t N = (512 * 1024 * 1024);
constexpr size_t X = (1024);

template <typename T>
using VectorAllocator = AlignedAllocator<T>;

template <typename T>
using AlignedVector = std::vector<T, VectorAllocator<T> >;

// Number of repetitions
constexpr int repetitions = 16;
constexpr int warm_up_token = -1;
// expected value of sum
int sum_expected = N-3;

static auto exception_handler = [](sycl::exception_list eList) {
  for (std::exception_ptr const &e : eList) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
      std::cout << "Failure" << std::endl;
      std::terminate();
    }
  }
};

class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

void flush_cache(sycl::queue &q, sycl::buffer<int> &flush_buf) {
  auto flush_size = flush_buf.get_size()/sizeof(int);
  auto ev = q.submit([&](auto &h) {
    sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::noinit);
    h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
  });
  ev.wait_and_throw();
}

int ComputeSerial(AlignedVector<int> &data, int iter) {
  const size_t data_size = data.size();
  Timer timer;
  int sum;
  // ComputeSerial main begin
  for (int it = 0; it < iter; it++) {
    sum = 0;
    for (size_t i = 0; i < data_size; ++i) {
      sum += data[i];
    }
  }
  // ComputeSerial main end
  double elapsed = timer.Elapsed();
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeSerial   = " << elapsed << "s"
              << " sum = " << sum << std::endl;
  else 
    std::cout << "ERROR: ComputeSerial Expected " << sum_expected << " but got "
              << sum << std::endl;
  return sum;
} // end ComputeSerial

void reductionAtomics1(sycl::queue &q, sycl::buffer<int> inbuf,
                     sycl::buffer<int> flush_buf, int &res, int iter) {
  const size_t data_size = inbuf.get_size()/sizeof(int);

  sycl::buffer<int> sum_buf(&res, 1);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);

      h.parallel_for(1, [=](auto index) {
        sum_acc[0] = 0;
      });
    });

    flush_cache(q, flush_buf);
    Timer timer;
    // reductionAtomics1 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(inbuf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);

      h.parallel_for(data_size, [=](auto index) {
        size_t glob_id = index[0];
        auto v =
            sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::acq_rel,
                                     sycl::ONEAPI::memory_scope::device,
                                     sycl::access::address_space::global_space>(
                sum_acc[0]);
        v.fetch_add(buf_acc[glob_id]);
      });
      // reductionAtomics1 main end
    });
    q.wait();
    {
      // ensure limited life-time of host accessor since it blocks the queue
      sycl::host_accessor h_acc(sum_buf);
      res = h_acc[0];
    }
    // do not measure time of warm-up iteration to exclude JIT compilation time
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  if (res == sum_expected)
    std::cout << "SUCCESS: Time reductionAtomics1   = " << elapsed << "s"
              << " sum = " << res << std::endl;
  else
    std::cout << "ERROR: reductionAtomics1 Expected " << sum_expected
              << " but got " << res << std::endl;
} // end reductionAtomics1

void reductionAtomics2(sycl::queue &q, sycl::buffer<int> inbuf,
                     sycl::buffer<int> flush_buf, int &res, int iter) {
  const size_t data_size = inbuf.get_size()/sizeof(int);

  int num_work_items = 1024 * 512;
  int BATCH = (N + num_work_items - 1) / num_work_items;
  std::cout << "Num work items = " << num_work_items << std::endl;

  sycl::buffer<int> sum_buf(&res,1);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; i++) {
    // init the acummulator on device
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(1,
                     [=](auto index) { sum_acc[0] = 0; });
    });

    flush_cache(q, flush_buf);
    Timer timer;
    //  reductionAtomics2 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(inbuf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(num_work_items, [=](auto index) {
        size_t glob_id = index[0];
        size_t start = glob_id * BATCH;
        size_t end = (glob_id + 1) * BATCH;
        if (end > N)
          end = N;
        int sum = 0;
        for (size_t i = start; i < end; i++)
          sum += buf_acc[i];
        auto v = 
	       sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::relaxed,
               sycl::ONEAPI::memory_scope::device,
               sycl::access::address_space::global_space>(sum_acc[0]);
        v.fetch_add(sum);
      });
    });
    // reductionAtomics2 main end
    q.wait();
    {
      sycl::host_accessor h_acc(sum_buf);
      res=h_acc[0];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  if (res == sum_expected)
    std::cout << "SUCCESS: Time reductionAtomics2   = " << elapsed << "s"
              << " sum = " << res << std::endl;
  else
    std::cout << "ERROR: reductionAtomics2 Expected " << sum_expected
              << " but got " << res << std::endl;
} // end reductionAtomics2

void reductionAtomics3(sycl::queue &q, sycl::buffer<int> inbuf,
                     sycl::buffer<int> flush_buf, int &res, int iter) {
  const size_t data_size = inbuf.get_size()/sizeof(int);

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  int num_work_items = 1024 * 1024;

  sycl::buffer<int> sum_buf(&res,1);
  std::cout << "starting\n";
  double elapsed = 0;
  for (int i = warm_up_token; i < iter; i++) {
    // init the acummulator on device
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(1,
                     [=](auto index) { sum_acc[0] = 0; });
    });

    flush_cache(q, flush_buf);
    Timer timer;
    // reductionAtomics3 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(inbuf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(num_work_items, [=](auto index) {
        size_t glob_id = index[0];
        int sum = 0;
        for (size_t i = glob_id; i < N; i+=num_work_items)
          sum += buf_acc[i];
        auto v = 
	       sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::relaxed,
               sycl::ONEAPI::memory_scope::device,
               sycl::access::address_space::global_space>(sum_acc[0]);
        v.fetch_add(sum);
      });
    });
    // reductionAtomics3 main end
    q.wait();
    {
      sycl::host_accessor h_acc(sum_buf);
      res=h_acc[0];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  if (res == sum_expected)
    std::cout << "SUCCESS: Time reductionAtomics3   = " << elapsed << "s"
              << " sum = " << res << std::endl;
  else
    std::cout << "ERROR: reductionAtomics3 Expected " << sum_expected
              << " but got " << res << std::endl;
} // end reductionAtomics3

void treeReduction(sycl::queue &q, sycl::buffer<int> inbuf,
                     sycl::buffer<int> flush_buf, int &res, int iter) {
  const size_t data_size = inbuf.get_size()/sizeof(int);

  int work_group_size = 
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  int num_work_items = data_size;
  int num_work_groups = num_work_items / work_group_size;

  sycl::buffer<int> sum_buf(&res, 1);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(1,
                     [=](auto index) { sum_acc[0] = 0; });
    });

    flush_cache(q, flush_buf);

    Timer timer;
    // treeReduction main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(inbuf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          scratch(work_group_size, h);

      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                     [=](sycl::nd_item<1> item) {
                       size_t global_id = item.get_global_id(0);
                       int local_id = item.get_local_id(0);
                       int group_id = item.get_group(0);

                       if (global_id < data_size)
                         scratch[local_id] = buf_acc[global_id];
                       else
                         scratch[local_id] = 0;

                       // Do a tree reduction on items in work-group
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (local_id < i)
                           scratch[local_id] += scratch[local_id + i];
                       }

                       if (local_id == 0) {
		           auto v = 
                             sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::relaxed,
                                  sycl::ONEAPI::memory_scope::device,
                                  sycl::access::address_space::global_space>(
                                  sum_acc[0]);
                           v.fetch_add(scratch[0]);
		       }
                     });
    });
    // treeReduction main end
    q.wait();
    {
      sycl::host_accessor h_acc(sum_buf);
      res = h_acc[0];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  if (res == sum_expected)
    std::cout << "SUCCESS: Time treeReduction   = " << elapsed << "s"
              << " sum = " << res << std::endl;
  else
    std::cout << "ERROR: treeReduction Expected " << sum_expected
              << " but got " << res << std::endl;
} // end treeReduction


void buf2bufReduction(sycl::queue &q, sycl::buffer<int> inbuf,
                              sycl::buffer<int> outbuf) {
  const size_t num_work_items = inbuf.get_size()/sizeof(int);

  int work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();

  q.submit([&](auto &h) {
         sycl::accessor buf_acc(inbuf, h, sycl::read_only);
         sycl::accessor outbuf_acc(outbuf, h, sycl::write_only, sycl::noinit);
         sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
             scratch(work_group_size, h);

         h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                     [=](sycl::nd_item<1> item) {
                       size_t global_id = item.get_global_id(0);
                       int local_id = item.get_local_id(0);
                       int group_id = item.get_group(0);

                       if (global_id < num_work_items)
                         scratch[local_id] = buf_acc[global_id];
                       else
                         scratch[local_id] = 0;

                       // Do a tree reduction on items in work-group
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (local_id < i)
                           scratch[local_id] += scratch[local_id + i];
                       }

                       if (local_id == 0)
                         outbuf_acc[group_id] = scratch[0];
                     });
    });
} // end buf2bufReduction

void buf2finalReduction(sycl::queue &q, sycl::buffer<int> inbuf,
                     int &res) {
  const size_t num_work_items = inbuf.get_size()/sizeof(int);

  int work_group_size = 
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  int num_work_groups = num_work_items / work_group_size;
  res=0;
  sycl::buffer<int> sum_buf(&res, 1);

    // buf2finalReduction main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(inbuf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          scratch(work_group_size, h);

      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                     [=](sycl::nd_item<1> item) {
                       size_t global_id = item.get_global_id(0);
                       int local_id = item.get_local_id(0);
                       int group_id = item.get_group(0);

                       if (global_id < num_work_items)
                         scratch[local_id] = buf_acc[global_id];
                       else
                         scratch[local_id] = 0;

                       // Do a tree reduction on items in work-group
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (local_id < i)
                           scratch[local_id] += scratch[local_id + i];
                       }

                       if (local_id == 0) {
		           auto v = 
                             sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::relaxed,
                                  sycl::ONEAPI::memory_scope::device,
                                  sycl::access::address_space::global_space>(
                                  sum_acc[0]);
                           v.fetch_add(scratch[0]);
		       }
                     });
    });
} // end buf2finalReduction

void builtinReduction(sycl::queue &q, sycl::buffer<int> inbuf,
                     sycl::buffer<int> flush_buf, int &res, int iter) {
  int num_work_items = inbuf.get_size()/sizeof(int);

  sycl::buffer<int> sum_buf(&res, 1);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(1, [=](auto index) { sum_acc[0] = 0; });
    });

    flush_cache(q, flush_buf);

    Timer timer;
    // builtinReduction main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(inbuf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::read_write);
      auto sumr = sycl::ONEAPI::reduction(sum_acc, sycl::ONEAPI::plus<>());
      h.parallel_for(sycl::nd_range<1>{num_work_items, 256}, sumr,
                     [=](sycl::nd_item<1> item, auto &sumr_arg) {
                       int glob_id = item.get_global_id(0);
                       sumr_arg += buf_acc[glob_id];
                     });
    });
    // builtinReduction main end
    q.wait();
    {
      sycl::host_accessor h_acc(sum_buf);
      res = h_acc[0];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  if (res == sum_expected)
    std::cout << "SUCCESS: Time builtinReduction   = " << elapsed << "s"
              << " sum = " << res << std::endl;
  else
    std::cout << "ERROR: builtinReduction Expected " << sum_expected
              << " but got " << res << std::endl;
} // end builtinReduction

void multiBlockInterleavedReduction(sycl::queue &q, sycl::buffer<int> inbuf,
                     sycl::buffer<int> flush_buf, int &res, int iter) {
  const size_t data_size = inbuf.get_size()/sizeof(int);
  const size_t flush_size = flush_buf.get_size();

  int work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  int elements_per_work_item = 256;
  int num_work_items = data_size / elements_per_work_item;
  int num_work_groups = num_work_items / work_group_size;

  std::cout << "Num work items = " << num_work_items << std::endl;
  std::cout << "Num work groups = " << num_work_groups << std::endl;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> sum_buf(&res, 1);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(1, [=](auto index) { sum_acc[0] = 0; });
    });

    flush_cache(q, flush_buf);

    Timer timer;
    // multiBlockInterleavedReduction1 main begin
    q.submit([&](auto &h) {
      const sycl::accessor buf_acc(inbuf, h);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(sycl::nd_range<1>{num_work_items, work_group_size}, 
	        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
            size_t glob_id = item.get_global_id(0);
            size_t group_id = item.get_group(0);
            size_t loc_id = item.get_local_id(0);
            sycl::ONEAPI::sub_group sg = item.get_sub_group();
	    size_t sg_size = sg.get_local_range()[0];
	    size_t sg_id = sg.get_group_id()[0];
            sycl::vec<int, 8> sum{0, 0, 0, 0, 0, 0, 0, 0};
            using global_ptr =
                sycl::multi_ptr<int, sycl::access::address_space::global_space>;
            int base = (group_id * work_group_size + sg_id * sg_size)
	                    * elements_per_work_item;
            for (size_t i = 0; i < elements_per_work_item / 8; i++)
              sum += sg.load<8>(global_ptr(&buf_acc[base + i * 8 * sg_size]));
	    int res=0;
            for (int i = 0; i < 8; i++)
               res += sum[i];
	    auto v = sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::relaxed,
                          sycl::ONEAPI::memory_scope::device,
                          sycl::access::address_space::global_space>(
                          sum_acc[0]);
            v.fetch_add(res);
          });
    });
    // multiBlockINterleavedReduction main end
    q.wait();
    {
      sycl::host_accessor h_acc(sum_buf);
      res = h_acc[0];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  if (res == sum_expected)
    std::cout << "SUCCESS: Time multiBLockInterleavedReduction   = " << elapsed << "s"
              << " sum = " << res << std::endl;
  else
    std::cout << "ERROR: multiBLockInterleavedReduction Expected " << sum_expected
              << " but got " << res << std::endl;
} // end multiBLockInterleavedReduction

void multiBlockInterleavedReductionVector(sycl::queue &q, sycl::buffer<int> inbuf,
                     sycl::buffer<int> flush_buf, int &res, int iter) {
  const size_t data_size = inbuf.get_size()/sizeof(int);

  int work_group_size = 
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  int num_work_items = data_size / 4;
  int num_work_groups = num_work_items / work_group_size;

  std::cout << "Num work items = " << num_work_items << std::endl;
  std::cout << "Num work groups = " << num_work_groups << std::endl;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> sum_buf(&res, 1);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      h.parallel_for(1, [=](auto index) { sum_acc[0] = 0; });
    });

    flush_cache(q, flush_buf);

    Timer timer;
    // MultiBLockedInterleavedReductionVector begin
    q.submit([&](auto &h) {
      const sycl::accessor buf_acc(inbuf, h);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          scratch(1, h);
      h.parallel_for(sycl::nd_range<1>{num_work_items, work_group_size}, 
	        [=](sycl::nd_item<1> item) {
            size_t glob_id = item.get_global_id(0);
            size_t group_id = item.get_group(0);
            size_t loc_id = item.get_local_id(0);
	    if (loc_id==0)
	       scratch[0]=0;
            sycl::vec<int, 4> val;
	    val.load(glob_id,buf_acc);
	    int sum=val[0]+val[1]+val[2]+val[3];

	    auto vl = sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::relaxed,
                          sycl::ONEAPI::memory_scope::work_group,
                          sycl::access::address_space::local_space>(
                          scratch[0]);
	    vl.fetch_add(sum);
            item.barrier(sycl::access::fence_space::local_space);
	    if (loc_id==0) {
	       auto v = sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::relaxed,
                          sycl::ONEAPI::memory_scope::device,
                          sycl::access::address_space::global_space>(
                          sum_acc[0]);
               v.fetch_add(scratch[0]);
	    }
          });
    });
    // MultiBLockedInterleavedReductionVector end
    q.wait();
    {
      sycl::host_accessor h_acc(sum_buf);
      res = h_acc[0];
    }
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  if (res == sum_expected)
    std::cout << "SUCCESS: MultiBLockedInterleavedReductionVector   = " << elapsed << "s"
              << " sum = " << res << std::endl;
  else
    std::cout << "ERROR: MultiBLockedInterleavedReductionVector Expected " << sum_expected
              << " but got " << res << std::endl;
}

int main(int argc, char *argv[]) {

  sycl::queue q{sycl::default_selector{}, exception_handler};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  VectorAllocator<int> alloc;
  AlignedVector<int> data(N, 1,alloc);
  AlignedVector<int> extra(N, 1,alloc);
  data[N/2+5467]=0;
  data[N-5]=0;
  data[5678]=0;
  int res=0;
  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> buf(data.data(), data.size(), props);
  sycl::buffer<int> flush_buf(extra.data(), extra.size(), props);
  ComputeSerial(data, 16);
  reductionAtomics1(q, buf, flush_buf, res, 16);
  reductionAtomics2(q, buf, flush_buf, res, 16);
  reductionAtomics3(q, buf, flush_buf, res, 16);
  treeReduction(q, buf, flush_buf, res, 16);
  builtinReduction(q, buf, flush_buf, res, 16);
  multiBlockInterleavedReduction(q, buf, flush_buf, res, 16);
  multiBlockInterleavedReductionVector(q, buf, flush_buf, res, 16);
}

