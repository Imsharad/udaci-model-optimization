<?xml version="1.0" encoding="UTF-8"?>
<rubric>
  <section id="baseline-model-analysis">
    <title>Baseline Model Analysis</title>
    <criteria>
      <item>
        <description>Establish a clear baseline for model performance metrics.</description>
        <requirements>The notebook displays execution results from each code block, including output from at least the provided performance metrics.</requirements>
      </item>
      <item>
        <description>Analyze the baseline model to identify optimization opportunities.</description>
        <requirements>The analysis report includes an evaluation of which compression techniques are likely to be most effective, with clear justification based on the model architecture and performance metrics.</requirements>
      </item>
    </criteria>
  </section>

  <section id="compression-techniques-implementation">
    <title>Compression Techniques Implementation</title>
    <criteria>
      <item>
        <description>Implement at least two compression techniques.</description>
        <requirements>
          Two or more optimization techniques are implemented, including:
          <list>
            <listitem>post-training (quantization, pruning, or graph optimization)</listitem>
            <listitem>in-training (quantization, pruning, or knowledge distillation) compression techniques.</listitem>
          </list>
          Each implementation includes appropriate parameter selection and training configuration and includes justification that explains why the implementation impacts model size, inference time and/or accuracy.
        </requirements>
      </item>
      <item>
        <description>Perform a comparative analysis of implemented compression techniques.</description>
        <requirements>The notebook includes a comparative analysis of all implemented compression techniques. The analysis includes comparisons across all three key metrics (size, speed, accuracy) and discusses the trade-offs between techniques. The student explains which techniques are most promising for a multi-stage pipeline and why.</requirements>
      </item>
    </criteria>
  </section>

  <section id="multi-stage-optimization-pipeline">
    <title>Multi-Stage Optimization Pipeline</title>
    <criteria>
      <item>
        <description>Design a relevant multi-stage optimization pipeline.</description>
        <requirements>The notebook includes an implementation plan with more than one pipeline design. Designs are given for one or more multi-stage optimization pipelines that combine at least two compression techniques, along with prioritizations of these pipelines. The pipeline design(s) includes a clear explanation of why specific techniques were chosen and sequenced in a particular order.</requirements>
      </item>
      <item>
        <description>Implement the multi-stage optimization pipeline.</description>
        <requirements>Code within the notebook implements the designed plan and the output in the notebook demonstrates the step-by-step results of execution. The implementation correctly applies at least one multi-step pipeline with steps executed in sequence and preserves intermediate models for evaluation. The student would ideally iterate on the implementation until CTO requirements are met.</requirements>
      </item>
      <item>
        <description>Report on how the multi-stage pipeline meets the optimization requirements.</description>
        <requirements>
          The notebook includes a report that outlines:
          <list>
            <listitem>whether the pipeline meets requirements</listitem>
            <listitem>review of experimentation done</listitem>
            <listitem>possible next steps for improvements</listitem>
          </list>
        </requirements>
      </item>
    </criteria>
  </section>

  <section id="mobile-deployment">
    <title>Mobile Deployment</title>
    <criteria>
      <item>
        <description>Convert the best optimized model for mobile.</description>
        <requirements>Code within the notebook converts the optimized model to a mobile-friendly format (PyTorch Mobile). The submission includes verification that the converted model maintains the expected performance characteristics.</requirements>
      </item>
      <item>
        <description>Provide a final analysis of the mobile deployment.</description>
        <requirements>
          The notebook includes analysis of mobile deployment considerations for the optimized model. The analysis includes:
          <list>
            <listitem>a discussion of use case-specific optimizations</listitem>
            <listitem>an overview of how (and why) to benchmark on mobile</listitem>
            <listitem>potential challenges</listitem>
            <listitem>future improvements</listitem>
          </list>
        </requirements>
      </item>
    </criteria>
  </section>

  <section id="final-reporting">
    <title>Final Reporting</title>
    <criteria>
      <item>
        <description>Create the technical section of the report.</description>
        <requirements>The submission includes a comprehensive technical report that documents the entire optimization process. The report includes well-structured sections covering baseline analysis, individual technique evaluations, pipeline design and implementation, and mobile deployment.</requirements>
      </item>
      <item>
        <description>Create the executive summary section of the report.</description>
        <requirements>
          The submission includes an executive summary that communicates how the solution meets business requirements and enables expansion to budget-friendly smartphones. It should include discussion of:
          <list>
            <listitem>user experience improvements</listitem>
            <listitem>market expansion opportunities</listitem>
            <listitem>how the technical achievements translate to business benefits</listitem>
          </list>
        </requirements>
      </item>
    </criteria>
  </section>

  <section id="suggestions">
    <title>Suggestions to Make Your Project Stand Out</title>
    <list>
      <listitem><emphasis>Extended Compression Techniques:</emphasis> Implement more than 2 techniques or even techniques beyond the standard options, such as neural architecture search, tensor decomposition, or low-rank factorization.</listitem>
      <listitem><emphasis>Hyperparameter Optimization:</emphasis> Implement systematic hyperparameter tuning (e.g., grid search, Bayesian optimization) for compression techniques to find optimal configurations.</listitem>
      <listitem><emphasis>Hardware-Specific Optimizations:</emphasis> Tailor optimizations for specific mobile hardware (ARM vs. x86, mobile GPUs) with benchmarks for different device profiles.</listitem>
      <listitem><emphasis>Comprehensive Reporting:</emphasis> Create a visualization dashboard that allows exploration of trade-offs between techniques and configurations.</listitem>
      <listitem><emphasis>Real-World Mobile Testing:</emphasis> Deploy the model on actual mobile devices and provide real-world performance measurements under various conditions.</listitem>
    </list>
  </section>
</rubric>