import { useCallback, useState, useEffect, useRef } from "react";
import {
  Eye,
  Scan,
  Info,
  AlertCircle,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { ImageUploader } from "@/components/upload/ImageUploader";
import { ImagePreview } from "@/components/upload/ImagePreview";
import { PredictionResults } from "@/components/analysis/PredictionResults";
import { GradCAMViewer } from "@/components/analysis/GradCAMViewer";
import { useImageUpload } from "@/hooks/useImageUpload";
import { usePrediction } from "@/hooks/usePrediction";
import { useHealthCheck } from "@/hooks/useHealthCheck";
import { getDisplayClassName } from "@/types/prediction";

export function AnalysisPage() {
  const {
    selectedImage,
    error: uploadError,
    selectImage,
    replaceImage,
    removeImage,
    setDimensions,
    setDecodeError,
  } = useImageUpload();

  const {
    status: predictionStatus,
    isLoading: isAnalyzing,
    isSuccess: hasPredictionResult,
    isError: hasPredictionError,
    result: predictionResult,
    error: predictionError,
    requestId,
    storageWarning,
    runAnalysis,
    updateAlpha,
    resetPrediction,
  } = usePrediction();

  const { modelStatus } = useHealthCheck();

  // Screen reader announcement state for aria-live region
  const [announcement, setAnnouncement] = useState<string>("");
  const prevStatusRef = useRef(predictionStatus);
  const prevAnalyzingRef = useRef(isAnalyzing);

  useEffect(() => {
    // Announce analysis start
    if (!prevAnalyzingRef.current && isAnalyzing) {
      setAnnouncement("Radiograph analysis in progress. Executing DenseNet121 inference and Grad-CAM backpropagation.");
    }
    // Announce analysis completion with class and confidence
    else if (prevAnalyzingRef.current && !isAnalyzing && hasPredictionResult && predictionResult) {
      const displayClass = getDisplayClassName(predictionResult.predicted_class);
      const confPercent = (predictionResult.confidence * 100).toFixed(2);
      setAnnouncement(`Analysis complete. Predicted category: ${displayClass} with ${confPercent}% confidence.`);
    }
    // Announce analysis failure
    else if (prevAnalyzingRef.current && !isAnalyzing && hasPredictionError) {
      setAnnouncement(`Analysis failed: ${predictionError || "An error occurred during inference."}`);
    }

    prevStatusRef.current = predictionStatus;
    prevAnalyzingRef.current = isAnalyzing;
  }, [isAnalyzing, hasPredictionResult, hasPredictionError, predictionResult, predictionError, predictionStatus]);

  // Reset prediction state whenever the user replaces or removes an image
  const handleFileSelect = useCallback(
    (file: File) => {
      resetPrediction();
      selectImage(file);
      setAnnouncement(`Radiograph staged: ${file.name}. Ready for analysis.`);
    },
    [resetPrediction, selectImage],
  );

  const handleFileReplace = useCallback(
    (file: File) => {
      resetPrediction();
      replaceImage(file);
      setAnnouncement(`Radiograph replaced with: ${file.name}. Ready for analysis.`);
    },
    [resetPrediction, replaceImage],
  );

  const handleFileRemove = useCallback(() => {
    resetPrediction();
    removeImage();
    setAnnouncement("Staged radiograph removed.");
  }, [resetPrediction, removeImage]);

  // Handle analysis trigger
  const handleStartAnalysis = async () => {
    if (!selectedImage) return;
    await runAnalysis(selectedImage.file);
  };

  // F-01: Handle interactive alpha adjustment locally without triggering backend re-inference
  const handleAlphaChange = (newAlpha: number) => {
    updateAlpha(newAlpha);
  };

  // F-01: Handle interactive target class switching as explainability refresh without duplicate scan history
  const handleTargetClassChange = async (targetClass: string) => {
    if (!selectedImage) return;
    await runAnalysis(selectedImage.file, { targetClass, isRefresh: true });
  };

  return (
    <div className="space-y-6">
      {/* Screen Reader Live Region for Analysis Announcements */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {announcement}
      </div>

      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            Chest Radiograph Analysis & Explainability
          </h2>
          <p className="text-slate-500 text-sm mt-0.5">
            Four-class DenseNet121 classification category inference and
            Grad-CAM visual activation heatmaps.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="clinical" size="md">
            Phase 3.5 • Full Pipeline Active
          </Badge>
        </div>
      </div>

      {/* Mandatory Medical Safety Directive */}
      <div className="p-3.5 rounded-xl border border-amber-200/80 bg-amber-50/60 text-amber-900 text-xs flex items-start gap-2.5">
        <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
        <div className="flex-1 leading-relaxed">
          <span className="font-semibold">Academic & Research Prototype:</span>{" "}
          This system is intended exclusively for technical research,
          benchmarking, and machine learning explainability demonstration. It
          does not provide medical advice or confirmed diagnoses, and must not
          guide clinical patient management.
        </div>
      </div>

      {/* Main Analysis Workflow Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Image Ingestion, Preview & Execution Controls (5 cols on lg) */}
        <div className="lg:col-span-5 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>
                {selectedImage ? "Radiograph Input" : "Image Ingestion"}
              </CardTitle>
              <CardDescription>
                {selectedImage
                  ? "Radiograph staged for DenseNet121 inference."
                  : "Accepts JPEG, PNG, and WEBP chest radiographs up to 10 MB."}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!selectedImage ? (
                <>
                  <ImageUploader
                    onFileSelect={handleFileSelect}
                    error={uploadError}
                  />

                  {/* Technical Specifications */}
                  <div className="rounded-lg bg-slate-50 p-3.5 border border-slate-100 text-[11px] text-slate-500 space-y-1.5">
                    <div className="font-semibold text-slate-700 text-xs mb-1">
                      Ingestion Specifications
                    </div>
                    <div>• Supported formats: JPEG, PNG, WEBP</div>
                    <div>• Maximum payload: 10 MB (10,485,760 bytes)</div>
                    <div>
                      • Projection: Posteroanterior (PA) or Anteroposterior (AP)
                    </div>
                    <div>
                      • Standardized input: 224 × 224 RGB via Lanczos filter
                    </div>
                    <div>• Preprocessing: DenseNet ImageNet normalization</div>
                  </div>
                </>
              ) : (
                <>
                  <ImagePreview
                    image={selectedImage}
                    onRemove={handleFileRemove}
                    onReplace={handleFileReplace}
                    onDimensionsLoaded={setDimensions}
                    onDecodeError={setDecodeError}
                  />

                  {/* Analysis Trigger Action Area */}
                  <div className="pt-2 border-t border-slate-100 space-y-3">
                    {/* Error Notice during Analysis */}
                    {hasPredictionError && (
                      <div className="p-3.5 rounded-lg border border-rose-200 bg-rose-50/80 text-rose-800 text-xs space-y-1.5">
                        <div className="flex items-start gap-2">
                          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                          <div className="flex-1 font-semibold">
                            Analysis Request Failed
                          </div>
                        </div>
                        <p className="text-[11px] leading-relaxed pl-6">
                          {predictionError}
                        </p>
                        {requestId && (
                          <div className="pl-6 font-mono text-[10px] text-rose-600">
                            Request ID: {requestId}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Non-blocking Local Storage Warning */}
                    {storageWarning && (
                      <div className="p-3 rounded-lg border border-amber-200 bg-amber-50 text-amber-800 text-xs flex items-start gap-2">
                        <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                        <div className="flex-1 leading-relaxed">
                          <span className="font-semibold">Storage Notice: </span>
                          {storageWarning}
                        </div>
                      </div>
                    )}

                    <Button
                      variant="primary"
                      size="md"
                      onClick={handleStartAnalysis}
                      disabled={isAnalyzing}
                      isLoading={isAnalyzing}
                      className="w-full text-xs font-semibold shadow-xs"
                      leftIcon={
                        !isAnalyzing ? <Scan className="w-4 h-4" /> : undefined
                      }
                    >
                      {isAnalyzing
                        ? "Processing Inference & Grad-CAM..."
                        : hasPredictionResult
                          ? "Re-run Analysis"
                          : hasPredictionError
                            ? "Retry Analysis"
                            : "Run Analysis"}
                    </Button>

                    <div className="flex items-center justify-between text-[11px] text-slate-500">
                      <span>
                        Model status:{" "}
                        <span className="font-medium text-slate-600">
                          {modelStatus === "ready" ? "Ready" : modelStatus}
                        </span>
                      </span>
                      <span>Target: conv5_block16_2</span>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Visualization & Prediction Results (7 cols on lg) */}
        <div className="lg:col-span-7 space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <div>
                <CardTitle>Visual Activation & Heatmap Overlay</CardTitle>
                <CardDescription>
                  DenseNet121 gradient backpropagation and classification
                  predictions.
                </CardDescription>
              </div>
              {hasPredictionResult && (
                <div className="flex items-center gap-1.5">
                  <Badge
                    variant="clinical"
                    size="sm"
                    className="font-mono text-xs"
                  >
                    Layer: conv5_block16_2
                  </Badge>
                </div>
              )}
            </CardHeader>
            <CardContent className="space-y-4">
              {/* State 1: No Image Selected Yet */}
              {!selectedImage && (
                <div className="h-80 rounded-xl bg-slate-50/80 border border-slate-200/80 flex flex-col items-center justify-center text-center p-6 text-slate-500">
                  <div className="w-14 h-14 rounded-2xl bg-slate-100 flex items-center justify-center mb-3 text-slate-500">
                    <Eye className="w-7 h-7" />
                  </div>
                  <p className="text-sm font-semibold text-slate-700">
                    No Radiograph Loaded
                  </p>
                  <p className="text-xs text-slate-500 mt-1 max-w-sm leading-relaxed">
                    Upload a chest radiograph on the left to stage input tensors
                    and compute Grad-CAM activation heatmaps.
                  </p>
                </div>
              )}

              {/* State 2: Image Selected, Pending Analysis Trigger */}
              {selectedImage && predictionStatus === "idle" && (
                <div className="h-80 rounded-xl bg-slate-50/80 border border-slate-200/80 flex flex-col items-center justify-center text-center p-6 text-slate-500 space-y-2">
                  <div className="w-14 h-14 rounded-2xl bg-clinical-50 text-clinical-600 flex items-center justify-center mb-1">
                    <Sparkles className="w-7 h-7" />
                  </div>
                  <h4 className="text-sm font-semibold text-slate-800">
                    Radiograph Staged & Verified
                  </h4>
                  <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                    Click{" "}
                    <span className="font-semibold text-slate-700">
                      "Run Analysis"
                    </span>{" "}
                    on the left to transmit image bytes to the backend inference
                    coordinator and compute Grad-CAM visual heatmaps.
                  </p>
                </div>
              )}

              {/* State 3: Active Inference Loading State */}
              {isAnalyzing && (
                <div className="h-80 rounded-xl bg-slate-950 border border-slate-800 flex flex-col items-center justify-center text-center p-6 text-white space-y-3">
                  <div className="relative">
                    <div className="w-16 h-16 rounded-full border-2 border-clinical-500/30 border-t-clinical-500 animate-spin motion-reduce:animate-none flex items-center justify-center" />
                    <Scan className="w-6 h-6 text-clinical-400 absolute inset-0 m-auto animate-pulse motion-reduce:animate-none" />
                  </div>
                  <div className="space-y-1">
                    <h5 className="text-sm font-semibold text-white">
                      Executing Deep Learning Inference
                    </h5>
                    <p className="text-xs text-slate-400 max-w-xs leading-relaxed">
                      Resampling to 224 × 224 RGB, computing DenseNet121 logits,
                      and backpropagating gradients at layer{" "}
                      <span className="font-mono text-clinical-300">
                        conv5_block16_2
                      </span>
                      ...
                    </p>
                  </div>
                </div>
              )}

              {/* State 4: Successful Prediction & Grad-CAM Visualization */}
              {hasPredictionResult && predictionResult && selectedImage && (
                <div className="space-y-6">
                  {/* Visual Explainability Heatmap & Overlay */}
                  <GradCAMViewer
                    result={predictionResult}
                    originalImageUrl={selectedImage.previewUrl}
                    onAlphaChange={handleAlphaChange}
                    onTargetClassChange={handleTargetClassChange}
                    isLoading={isAnalyzing}
                  />

                  {/* Four-Class Softmax Probabilities & Classification Results */}
                  <PredictionResults result={predictionResult} />
                </div>
              )}

              {/* State 5: Analysis Error State */}
              {selectedImage && hasPredictionError && !isAnalyzing && (
                <div className="h-72 rounded-xl bg-slate-50 border border-rose-200/80 flex flex-col items-center justify-center text-center p-6 text-slate-600 space-y-3">
                  <AlertCircle className="w-10 h-10 text-rose-500" />
                  <div className="space-y-1">
                    <h5 className="text-sm font-semibold text-slate-900">
                      Inference Failed
                    </h5>
                    <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                      {predictionError}
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleStartAnalysis}
                    className="text-xs bg-white mt-1"
                    leftIcon={
                      <RefreshCw className="w-3.5 h-3.5 text-slate-600" />
                    }
                  >
                    Retry Analysis
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default AnalysisPage;
