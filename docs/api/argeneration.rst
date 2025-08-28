ARGeneration API
================

The ARGeneration class is an abstract base class for autoregressive generation models in MLLM. It provides essential functionalities for generating sequences using various sampling methods and tracing capabilities.

.. code-block:: cpp

   #include "mllm/models/ARGeneration.hpp"

Base Class
----------

.. cpp:class:: ARGeneration

   Abstract base class for autoregressive generation models.

Protected Attributes
--------------------

.. cpp:member:: bool ARGeneration::do_sample_

   Flag indicating whether to perform sampling during generation. Default is false.

.. cpp:member:: int ARGeneration::eos_token_id_

   End-of-sequence token ID used to terminate generation.

.. cpp:member:: int ARGeneration::max_length_

   Maximum length of generated sequences. Default is 1024.

Core Virtual Methods
--------------------

.. cpp:function:: virtual ARGenerationOutputPast ARGeneration::forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) = 0

   Pure virtual function for forward pass of the model.

   :param input: Input tensors map
   :param args: Arguments for the forward pass
   :return: Output tensors map with past states

.. cpp:function:: virtual ARGenerationOutputPast ARGeneration::generate(const ARGenerationOutputPast& input, const ARGenerationArgs& args)

   Generate sequences using the model.

   :param input: Input tensors map
   :param args: Arguments for generation
   :return: Generated output tensors map with past states

.. cpp:function:: virtual void ARGeneration::streamGenerate(const ARGenerationOutputPast& input, const ARGenerationArgs& args, const std::function<void(int64_t)>& callback)

   Generate sequences with streaming output.

   :param input: Input tensors map
   :param args: Arguments for generation
   :param callback: Callback function to handle generated tokens

.. cpp:function:: virtual IROutput ARGeneration::trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args)

   Trace the model execution for compilation or analysis.

   :param input: Input tensors map
   :param args: Arguments for tracing
   :return: IR context output map

Sampling Methods
----------------

.. cpp:function:: int64_t ARGeneration::sampleGreedy(Tensor& logits)

   Sample the next token using greedy strategy (select the token with highest probability).

   :param logits: Logits tensor from the model
   :return: Selected token ID

.. cpp:function:: int64_t ARGeneration::sampleTemperature(Tensor& logits, float temperature)

   Sample the next token using temperature-based sampling.

   :param logits: Logits tensor from the model
   :param temperature: Temperature value for sampling (higher values increase randomness)
   :return: Selected token ID

.. cpp:function:: int64_t ARGeneration::sampleTopK(Tensor& logits, int k, float temperature)

   Sample the next token using top-k sampling strategy.

   :param logits: Logits tensor from the model
   :param k: Number of top tokens to consider
   :param temperature: Temperature value for sampling
   :return: Selected token ID

.. cpp:function:: int64_t ARGeneration::sampleTopP(Tensor& logits, float p, float temperature)

   Sample the next token using nucleus (top-p) sampling strategy.

   :param logits: Logits tensor from the model
   :param p: Cumulative probability threshold
   :param temperature: Temperature value for sampling
   :return: Selected token ID

.. cpp:function:: int64_t ARGeneration::categoricalSample(const Tensor& probs)

   Sample from a categorical distribution.

   :param probs: Probability distribution tensor
   :return: Sampled token ID

Utility Methods
---------------

.. cpp:function:: Tensor ARGeneration::getLastLogits(Tensor& logits)

   Extract the logits for the last token in the sequence.

   :param logits: Full logits tensor
   :return: Logits for the last token

.. cpp:function:: int ARGeneration::sampleFromDistribution(const std::vector<float>& probs)

   Sample from a probability distribution.

   :param probs: Vector of probabilities
   :return: Sampled index