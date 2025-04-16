import boto3
import subprocess
import os
import logging
import json

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')  # Client to invoke Lambda functions

def handler(event, context):
    try:
        # Extract bucket and object information from the S3 event
        record = event['Records'][0]['s3']
        input_bucket = record['bucket']['name']
        object_key = record['object']['key']
        
        # Validate input file is an MP4
        if not object_key.lower().endswith('.mp4'):
            logger.error(f"Invalid file format: {object_key}. Only MP4 files are supported.")
            return {
                'statusCode': 400,
                'body': 'Invalid file format. Only MP4 files are supported.'
            }
        
        # Define paths
        video_name = os.path.splitext(object_key)[0]  # Remove .mp4 extension
        download_path = f"/tmp/{object_key}"
        output_file = f"/tmp/{video_name}.jpg"  # Single output frame
        
        # Download video from input bucket
        logger.info(f"Downloading video: {object_key}")
        s3_client.download_file(input_bucket, object_key, download_path)
        
        # Run ffmpeg command to extract a single frame
        ffmpeg_command = [
            "ffmpeg", "-ss", "0",  # Start from the beginning of the video
            "-i", download_path,  # Input video path
            "-vframes", "1",      # Extract only one frame
            output_file,          # Output frame file
            "-y"                  # Overwrite if the file exists
        ]
        
        logger.info("Running ffmpeg command")
        subprocess.run(ffmpeg_command, check=True)
        
        # Upload the extracted frame to the stage-1 bucket
        output_bucket = input_bucket.replace('-input', '-stage-1')
        frame_key = f"{video_name}.jpg"  # Flat structure
        
        logger.info(f"Uploading frame: {frame_key}")
        s3_client.upload_file(output_file, output_bucket, frame_key)

        # Invoke the face-recognition Lambda function
        face_recognition_payload = {
            "bucket_name": output_bucket,
            "image_file_name": frame_key
        }
        response = lambda_client.invoke(
            FunctionName="face-recognition",  # Ensure this matches your Lambda function's name
            InvocationType="Event",  # Asynchronous invocation
            Payload=json.dumps(face_recognition_payload)
        )
        logger.info(f"Invoked face-recognition Lambda for {frame_key}, Response: {response['StatusCode']}")
        
        # Cleanup temporary files
        subprocess.run(["rm", download_path])
        subprocess.run(["rm", output_file])
        
        return {
            'statusCode': 200,
            'body': f"Successfully processed video {object_key} into {output_bucket}/{frame_key} and invoked face-recognition Lambda."
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            'statusCode': 500,
            'body': f"Error processing video: {str(e)}"
        }
