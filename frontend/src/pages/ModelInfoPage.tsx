import {
  ShieldCheck,
  Layers,
  Database,
  BarChart3,
  Sliders,
  AlertTriangle,
  Info,
  Cpu,
} from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

export function ModelInfoPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            Model Architecture & Provenance Specifications
          </h2>
          <p className="text-slate-500 text-sm mt-0.5">
            Technical architecture, dataset partitions, empirical benchmarks, and Grad-CAM explainability mechanics.
          </p>
        </div>
        <Badge variant="clinical" size="md">
          DenseNet121 • Transfer Learning
        </Badge>
      </div>

      {/* Mandatory Medical Safety Disclaimer */}
      <div className="p-4 rounded-xl border border-amber-200/80 bg-amber-50/70 text-amber-900 text-xs flex items-start gap-3">
        <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
        <div className="space-y-1 leading-relaxed">
          <div className="font-semibold text-amber-950">
            Mandatory Academic Research & Technical Evaluation Disclaimer
          </div>
          <p>
            LungCare AI is an experimental computer-vision research prototype developed for academic and educational evaluation. It is <strong>not an FDA-cleared, CE-marked, or clinically validated diagnostic device</strong>. Model probability predictions and Grad-CAM visual heatmaps must never be used as a primary diagnostic tool, clinical decision aid, or substitute for professional radiological consultation.
          </p>
        </div>
      </div>

      {/* Section 1: Architecture Highlights Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center justify-between">
              <span>Backbone</span>
              <Cpu className="w-4 h-4 text-clinical-600" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-slate-900">DenseNet121</div>
            <p className="text-xs text-slate-500 mt-1">
              Densely Connected Convolutional Network with ImageNet weights.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center justify-between">
              <span>Input Tensor</span>
              <Layers className="w-4 h-4 text-clinical-600" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-slate-900">224 × 224 × 3</div>
            <p className="text-xs text-slate-500 mt-1">
              RGB float32 via TensorFlow/Keras DenseNet121 preprocessing using <code className="font-mono text-[10px] bg-slate-100 text-slate-700 px-1 py-0.5 rounded">tf.keras.applications.densenet.preprocess_input</code>.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center justify-between">
              <span>Explainability Layer</span>
              <Sliders className="w-4 h-4 text-clinical-600" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-base font-bold text-slate-900 font-mono truncate">
              conv5_block16_2_conv
            </div>
            <p className="text-xs text-slate-500 mt-1">
              32-channel bottleneck convolution evaluated via tf.GradientTape.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center justify-between">
              <span>Classification Head</span>
              <ShieldCheck className="w-4 h-4 text-clinical-600" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-slate-900">4 Softmax Units</div>
            <p className="text-xs text-slate-500 mt-1">
              GAP (1024-d) &rarr; BatchNorm &rarr; Dropout (0.4) &rarr; Dense.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Section 2: Detailed Architecture Pipeline */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-clinical-600" />
            <span>Neural Network Graph & Transfer Learning Head</span>
          </CardTitle>
          <CardDescription>
            Architectural configuration of the deployed Keras/TensorFlow model artifact.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl border border-slate-200/80 bg-slate-50/50 space-y-2 text-xs">
              <h5 className="font-semibold text-slate-800 text-sm">Feature Extraction Backbone</h5>
              <p className="text-slate-600 leading-relaxed">
                The feature extraction pipeline employs <strong>DenseNet121</strong> pre-trained on ImageNet. Dense connectivity connects each layer to every other layer in a feed-forward fashion, preserving feature reuse across dense blocks and alleviating vanishing gradients.
              </p>
              <ul className="space-y-1 text-slate-600 list-disc pl-4 pt-1">
                <li>Input standardization: Resampled to 224 × 224 pixels using Lanczos interpolation.</li>
                <li>Preprocessing: TensorFlow/Keras DenseNet121 preprocessing using <code className="font-mono text-slate-800 bg-slate-100 px-1 py-0.5 rounded">tf.keras.applications.densenet.preprocess_input</code>.</li>
                <li>Final dense block: Block 4 with 16 dual-convolution sub-blocks terminating in layer <code className="font-mono text-slate-800 bg-slate-100 px-1 py-0.5 rounded">conv5_block16_2_conv</code>.</li>
              </ul>
            </div>

            <div className="p-4 rounded-xl border border-slate-200/80 bg-slate-50/50 space-y-2 text-xs">
              <h5 className="font-semibold text-slate-800 text-sm">Custom Classification Head</h5>
              <p className="text-slate-600 leading-relaxed">
                The top classification head adapts the high-level 1024-dimensional feature representations to the four pulmonary categories using regularization to mitigate overfitting:
              </p>
              <ul className="space-y-1 text-slate-600 list-disc pl-4 pt-1">
                <li><strong>GlobalAveragePooling2D:</strong> Collapses spatial 7 × 7 feature maps into a 1024-dimensional vector.</li>
                <li><strong>BatchNormalization:</strong> Stabilizes activation distributions before dense projections.</li>
                <li><strong>Dropout (rate = 0.4):</strong> Randomly drops 40% of activations during training to encourage distributed representation.</li>
                <li><strong>Dense Layer (4 units, Softmax):</strong> Produces categorical probability distribution $S_c \in [0.0, 1.0]$.</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Section 3: Dataset Provenance & Split Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <Database className="w-4 h-4 text-clinical-600" />
            <span>Dataset Provenance & Partition Integrity</span>
          </CardTitle>
          <CardDescription>
            Dataset composition from Kaggle research source jtiptj/chest-xray-pneumoniacovid19tuberculosis.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
            <div className="p-3.5 rounded-xl border border-slate-100 bg-slate-50">
              <div className="text-xs text-slate-500">Original Training Set</div>
              <div className="text-xl font-bold text-slate-900 font-mono mt-0.5">6,326</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Source images</div>
            </div>
            <div className="p-3.5 rounded-xl border border-slate-100 bg-slate-50">
              <div className="text-xs text-slate-500">Final Train Split</div>
              <div className="text-xl font-bold text-clinical-700 font-mono mt-0.5">5,377</div>
              <div className="text-[10px] text-slate-400 mt-0.5">85% partition</div>
            </div>
            <div className="p-3.5 rounded-xl border border-slate-100 bg-slate-50">
              <div className="text-xs text-slate-500">Stratified Val Split</div>
              <div className="text-xl font-bold text-clinical-700 font-mono mt-0.5">949</div>
              <div className="text-[10px] text-slate-400 mt-0.5">15% balanced holdout</div>
            </div>
            <div className="p-3.5 rounded-xl border border-slate-100 bg-slate-50">
              <div className="text-xs text-slate-500">Original Test Set</div>
              <div className="text-xl font-bold text-slate-900 font-mono mt-0.5">771</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Separate Kaggle test fold</div>
            </div>
          </div>

          <div className="p-3.5 rounded-lg border border-amber-200/80 bg-amber-50/60 text-xs text-amber-900 space-y-1.5 leading-relaxed">
            <div className="font-semibold text-amber-950 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-700 shrink-0" />
              <span>Dataset Split Integrity & Data Leakage Notice:</span>
            </div>
            <p>
              1. <strong>Validation Creation:</strong> The 949-sample validation split was generated directly from the original 6,326 training images using stratified sampling across the four classes.
            </p>
            <p>
              2. <strong>Identified Duplicates:</strong> During repository audit, <strong>3 duplicate images</strong> were identified between the original training and validation folders.
            </p>
            <p>
              3. <strong>Image-Level Splitting:</strong> The source dataset splits were performed at the <em>image level</em> rather than <em>patient level</em> (due to absence of patient metadata in the public dataset). Consequently, patient-level cross-split data leakage cannot be ruled out, which may artificially elevate academic validation metrics compared to real-world clinical distributions.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Section 4: Rigorous Separation of Evaluation Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-clinical-600" />
            <span>Documented Evaluation Benchmarks</span>
          </CardTitle>
          <CardDescription>
            Strictly distinguished validation and baseline test evaluation metrics from lungcare_evaluation.json.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Benchmark 1: Stratified Validation Holdout */}
            <div className="p-4 rounded-xl border border-clinical-200/80 bg-clinical-50/40 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-clinical-900">
                  Stratified Validation Set
                </span>
                <Badge variant="clinical" size="sm">
                  949 Samples
                </Badge>
              </div>

              <div className="space-y-1">
                <div className="text-3xl font-extrabold text-clinical-950 font-mono">
                  97.68%
                </div>
                <div className="text-xs text-clinical-800 font-medium">
                  Validation Accuracy (927 / 949 correct)
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-clinical-200/60 font-mono">
                <div>
                  <span className="text-slate-500 block text-[11px]">Weighted F1:</span>
                  <span className="font-semibold text-slate-900">97.69%</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px]">Macro F1:</span>
                  <span className="font-semibold text-slate-900">97.34%</span>
                </div>
              </div>

              <p className="text-[11px] text-slate-600 leading-relaxed pt-1">
                Evaluated on the internal holdout split derived from the training set. Reflects strong statistical fit to the research distribution.
              </p>
            </div>

            {/* Benchmark 2: Aggregate Metrics from Evaluation Artifact */}
            <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/60 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                  Validation Aggregate Metrics
                </span>
                <Badge variant="outline" size="sm">
                  Artifact Verified
                </Badge>
              </div>

              <div className="space-y-1">
                <div className="text-2xl font-extrabold text-slate-900 font-mono">
                  97.34% <span className="text-sm font-normal text-slate-500">Macro</span> / 97.69% <span className="text-sm font-normal text-slate-500">Weighted</span>
                </div>
                <div className="text-xs text-slate-600 font-medium">
                  Macro & Weighted F1-Scores (lungcare_evaluation.json)
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-200 font-mono">
                <div>
                  <span className="text-slate-500 block text-[11px]">Macro Avg (P / R):</span>
                  <span className="font-semibold text-slate-900">96.70% / 98.01%</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px]">Weighted Avg (P / R):</span>
                  <span className="font-semibold text-slate-900">97.73% / 97.68%</span>
                </div>
              </div>

              <p className="text-[11px] text-slate-600 leading-relaxed pt-1">
                Metrics directly sourced from <code className="font-mono bg-white px-1 py-0.5 rounded border border-slate-200 text-slate-800">lungcare_evaluation.json</code>. The evaluation artifact documents the 949-sample validation holdout; test partition metrics are omitted as they are not recorded in this artifact.
              </p>
            </div>
          </div>

          {/* Per-Class Validation Classification Report Table */}
          <div className="space-y-2">
            <h5 className="text-xs font-semibold text-slate-800 uppercase tracking-wider">
              Stratified Validation Breakdown by Category (949 Samples)
            </h5>
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table
                className="w-full text-xs text-left"
                aria-label="Stratified validation performance breakdown by category"
              >
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold">
                  <tr>
                    <th scope="col" className="py-2.5 px-3">Class Label</th>
                    <th scope="col" className="py-2.5 px-3">Precision</th>
                    <th scope="col" className="py-2.5 px-3">Recall</th>
                    <th scope="col" className="py-2.5 px-3">F1-Score</th>
                    <th scope="col" className="py-2.5 px-3 text-right">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
                  <tr>
                    <th scope="row" className="py-2 px-3 font-sans font-medium text-slate-900 text-left">COVID-19</th>
                    <td className="py-2 px-3">95.71%</td>
                    <td className="py-2 px-3">97.10%</td>
                    <td className="py-2 px-3 font-semibold text-clinical-700">96.40%</td>
                    <td className="py-2 px-3 text-right">69</td>
                  </tr>
                  <tr>
                    <th scope="row" className="py-2 px-3 font-sans font-medium text-slate-900 text-left">Normal</th>
                    <td className="py-2 px-3">93.78%</td>
                    <td className="py-2 px-3">97.51%</td>
                    <td className="py-2 px-3 font-semibold text-emerald-700">95.61%</td>
                    <td className="py-2 px-3 text-right">201</td>
                  </tr>
                  <tr>
                    <th scope="row" className="py-2 px-3 font-sans font-medium text-slate-900 text-left">Pneumonia</th>
                    <td className="py-2 px-3">99.30%</td>
                    <td className="py-2 px-3">97.42%</td>
                    <td className="py-2 px-3 font-semibold text-amber-700">98.35%</td>
                    <td className="py-2 px-3 text-right">581</td>
                  </tr>
                  <tr>
                    <th scope="row" className="py-2 px-3 font-sans font-medium text-slate-900 text-left">Tuberculosis</th>
                    <td className="py-2 px-3">98.00%</td>
                    <td className="py-2 px-3">100.00%</td>
                    <td className="py-2 px-3 font-semibold text-rose-700">98.99%</td>
                    <td className="py-2 px-3 text-right">98</td>
                  </tr>
                  <tr className="bg-slate-50/50 font-medium text-slate-800">
                    <th scope="row" className="py-2 px-3 font-sans text-left">Macro Average</th>
                    <td className="py-2 px-3">96.70%</td>
                    <td className="py-2 px-3">98.01%</td>
                    <td className="py-2 px-3 text-slate-900">97.34%</td>
                    <td className="py-2 px-3 text-right">949</td>
                  </tr>
                  <tr className="bg-slate-50/80 font-semibold text-slate-900">
                    <th scope="row" className="py-2 px-3 font-sans text-left">Weighted Average</th>
                    <td className="py-2 px-3">97.73%</td>
                    <td className="py-2 px-3">97.68%</td>
                    <td className="py-2 px-3 text-clinical-700">97.69%</td>
                    <td className="py-2 px-3 text-right">949</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Section 5: Grad-CAM Explainability Mechanics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-clinical-600" />
            <span>Grad-CAM Visual Explainability Mechanics</span>
          </CardTitle>
          <CardDescription>
            Gradient-weighted Class Activation Mapping methodology and interpretation boundaries.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-xs text-slate-600 leading-relaxed">
          <div className="p-4 rounded-xl border border-slate-200/80 bg-slate-50 space-y-2">
            <h5 className="font-semibold text-slate-800 text-sm">Target Convolutional Layer</h5>
            <p>
              Explainability gradients are computed against the final convolutional layer of the backbone: <code className="font-mono text-slate-800 bg-white px-1.5 py-0.5 rounded border border-slate-200">conv5_block16_2_conv</code>.
              This layer represents the highest-level spatial abstractions (receptive field covering pulmonary consolidation patterns) before global average pooling collapses spatial dimensions.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl border border-slate-200/80 bg-white space-y-2">
              <h5 className="font-semibold text-slate-800 text-sm">Why Explainability Matters</h5>
              <ul className="space-y-1 list-disc pl-4 text-slate-600">
                <li>Detects shortcut learning and confounding artifacts (e.g. radiological markers, tubing, edge glare).</li>
                <li>Provides visual verification of anatomical focus (verifying whether activations center on lung parenchyma vs. background).</li>
                <li>Fosters model transparency for machine learning research and peer review.</li>
              </ul>
            </div>

            <div className="p-4 rounded-xl border border-slate-200/80 bg-white space-y-2">
              <h5 className="font-semibold text-slate-800 text-sm">Critical Clinical Limitations</h5>
              <ul className="space-y-1 list-disc pl-4 text-slate-600">
                <li>Grad-CAM is <strong>not a lesion segmentation or localization tool</strong>.</li>
                <li>Heatmaps display visual feature correlations, not pathological disease boundaries.</li>
                <li>Coarse spatial resolution ($7 \times 7$ upsampled to $224 \times 224$) cannot demarcate fine interstitial patterns or micro-nodules.</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ModelInfoPage;
