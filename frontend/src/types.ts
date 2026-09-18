export type Health = { status: string; model_loaded: boolean; model_version: string | null }
export type ModelInfo = { model_version: string; model_type: string; trained_at: string | null; features: string[]; classes: string[]; sklearn_version: string | null; test_accuracy: number | null; n_training_samples: number | null }
export type Prediction = { prediction: string; confidence?: number; probability?: number; probabilities?: Record<string, number>; ranked?: { species: string; probability: number }[]; model_version: string; request_id: string }
export type PredictionInput = { sepal_length: number; sepal_width: number; petal_length: number; petal_width: number }
