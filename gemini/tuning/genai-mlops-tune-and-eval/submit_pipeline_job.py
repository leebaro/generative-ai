from google.cloud import aiplatform

aiplatform.init(
    project="[your-project-id]", location="us-central1"
)  # Initialize the Vertex AI SDK

job = aiplatform.PipelineJob(
<<<<<<< HEAD
    display_name="tuned-gemini-1-5-pro-002",  # Give your run a name
=======
    display_name="tuned-gemini-2.0-flash",  # Give your run a name
>>>>>>> main
    template_path="pipeline.json",  # Path to the compiled pipeline JSON
    pipeline_root="gs://vertex-ai-pipeline-root-20250116",  # Cloud Storage location for pipeline artifacts
    parameter_values={
        "project": "[your-project-id]",
        "location": "us-central1",
<<<<<<< HEAD
        "source_model_name": "gemini-1.5-pro-002",
=======
        "source_model_name": "gemini-2.0-flash",
>>>>>>> main
        "train_data_uri": "gs://github-repo/generative-ai/gemini/tuning/mlops-tune-and-eval/patient_1_glucose_examples.jsonl",
    },  # Pass pipeline parameter values here
)

job.run()  # Submit the pipeline for execution
