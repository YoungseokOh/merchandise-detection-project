# Import the InferencePipeline object
from inference import InferencePipeline
import cv2
import os

# Initialize a counter outside the function to keep track across frames
frame_count = 0

def my_sink(result, video_frame):
    global frame_count  # Use the global frame counter
    if result.get("output_image"):  # Check if there is an output image in the result
        # Define the save path with frame count for unique filenames
        save_path = os.path.join("E:\\merchandise_dataset\\workflow_results", f"workflow_image_{frame_count:04d}.jpg")
        cv2.imwrite(save_path, result["output_image"].numpy_image)
        frame_count += 1  # Increment the frame counter for each saved image
    print(result)  # Process or print predictions if needed


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="3eDw4J9aKw6I4hKjvGf1",
    workspace_name="merchandiseproject",
    workflow_id="detect-count-and-visualize-jch",
    video_reference="E:/merchandise_dataset/test_video.mp4", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish