// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_OPTIM_SAMPLER_H_
#define COLMAP_SRC_OPTIM_SAMPLER_H_

#include <cstddef>
#include <vector>
#include <iostream>
#include <iterator>
#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <algorithm>

using namespace boost::random;

#include "util/logging.h"

namespace colmap {

// Abstract base class for sampling methods.
class Sampler {
 public:
  Sampler(){};
  explicit Sampler(const size_t num_samples);

  // Initialize the sampler, before calling the `Sample` method.
  virtual void Initialize(const size_t total_num_samples) = 0;

  // Maximum number of unique samples that can be generated.
  virtual size_t MaxNumSamples() = 0;

  // Sample `num_samples` elements from all samples.
  virtual std::vector<size_t> Sample() = 0;

  // Sample elements from `X` into `X_rand`.
  //
  // Note that `X.size()` should equal `num_total_samples` and `X_rand.size()`
  // should equal `num_samples`.
  template <typename X_t>
  void SampleX(const X_t& X, X_t* X_rand);

  // Sample elements from `X` and `Y` into `X_rand` and `Y_rand`.
  //
  // Note that `X.size()` should equal `num_total_samples` and `X_rand.size()`
  // should equal `num_samples`. The same applies for `Y` and `Y_rand`.
  template <typename X_t, typename Y_t>
  void SampleXY(const X_t& X, const Y_t& Y, X_t* X_rand, Y_t* Y_rand);

  
  template <typename X_t, typename Y_t>
  void SampleXY_Weighted(const X_t& X, const Y_t& Y, X_t* X_rand, Y_t* Y_rand, std::vector<float> sample_weights);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename X_t>
void Sampler::SampleX(const X_t& X, X_t* X_rand) {
  const auto sample_idxs = Sample();
  for (size_t i = 0; i < X_rand->size(); ++i) {
    (*X_rand)[i] = X[sample_idxs[i]];
  }
}

template <typename X_t, typename Y_t>
void Sampler::SampleXY(const X_t& X, const Y_t& Y, X_t* X_rand, Y_t* Y_rand) {
  CHECK_EQ(X.size(), Y.size());
  CHECK_EQ(X_rand->size(), Y_rand->size());
  const auto sample_idxs = Sample();
  for (size_t i = 0; i < X_rand->size(); ++i) {
    (*X_rand)[i] = X[sample_idxs[i]];
    (*Y_rand)[i] = Y[sample_idxs[i]];
  }
}

template <typename X_t, typename Y_t>
void Sampler::SampleXY_Weighted(const X_t& X, const Y_t& Y, X_t* X_rand, Y_t* Y_rand, std::vector<float> sample_weights) 
{
  CHECK_EQ(X.size(), Y.size());
  CHECK_EQ(X.size(), sample_weights.size());
  CHECK_EQ(X_rand->size(), Y_rand->size());

  uniform_01<> dist;
  boost::random::mt19937 gen(342575235);
  std::vector<double> vals;
  
  //---------------------------------------
  //Sample Indices according to distribution. 
  //---------------------------------------
  for (auto iter : sample_weights) 
  {
    vals.push_back(std::pow(dist(gen), 1. / iter));
  }

  // Sorting vals, but retain the indices. 
  // There is unfortunately no easy way to do this with STL.
  std::vector<std::pair<int, double>> valsWithIndices;
  for (size_t iter = 0; iter < vals.size(); iter++) 
  {
    valsWithIndices.emplace_back(iter, vals[iter]);
  }
  
  std::sort(valsWithIndices.begin(), valsWithIndices.end(), [](auto x, auto y) {return x.second > y.second; });
  
  std::vector<int> sample_idxs;
  int sampleSize = X_rand->size();
  for (auto iter = 0; iter < sampleSize; iter++) 
  {
        sample_idxs.push_back(valsWithIndices[iter].first);
  }

  //Apply indices to get our random sampling. 
  for (size_t i = 0; i < X_rand->size(); ++i) 
  {
    (*X_rand)[i] = X[sample_idxs[i]];
    (*Y_rand)[i] = Y[sample_idxs[i]];
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_SAMPLER_H_
