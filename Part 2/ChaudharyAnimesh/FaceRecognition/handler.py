# import boto3
# import os
# import logging
# import json
# import torch
# import cv2
# from PIL import Image
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import numpy as np
# from botocore.exceptions import ClientError

# # Initialize logging
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# # Initialize S3 client
# s3 = boto3.client('s3')

# # Global variables for model instances
# global_vars = {
#     'mtcnn': None,
#     'resnet': None,
#     'saved_data': None,
#     'initialized': False
# }

# def download_if_not_exists(bucket, key, local_path):
#     """Download file if it doesn't exist locally"""
#     try:
#         if not os.path.exists(local_path):
#             logger.info(f"Downloading {key} from {bucket}")
#             s3.download_file(bucket, key, local_path)
#             logger.info(f"Download complete: {key}")
#         return True
#     except ClientError as e:
#         logger.error(f"Error downloading {key}: {str(e)}")
#         return False

# def load_models():
#     """Initialize models with better error handling"""
#     try:
#         if global_vars['initialized']:
#             return True

#         logger.info("Starting model initialization")
        
#         # Ensure we're using CPU
#         torch.set_grad_enabled(False)
        
#         # Download data.pt from your data bucket
#         data_success = download_if_not_exists('1229421130-data', 'data.pt', '/tmp/data.pt')
#         if not data_success:
#             raise Exception("Failed to download data.pt")
            
#         logger.info("Loading data.pt...")
#         global_vars['saved_data'] = torch.load('/tmp/data.pt', map_location='cpu')
        
#         logger.info("Initializing MTCNN...")
#         global_vars['mtcnn'] = MTCNN(
#             image_size=160,
#             margin=0,
#             min_face_size=20,
#             thresholds=[0.6, 0.7, 0.7],
#             factor=0.709,
#             post_process=True,
#             device='cpu'
#         )
        
#         logger.info("Initializing ResNet...")
#         global_vars['resnet'] = InceptionResnetV1(pretrained='vggface2').eval()
        
#         global_vars['initialized'] = True
#         logger.info("Model initialization completed successfully")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error in load_models: {str(e)}")
#         return False

# def process_face(image_path):
#     """Process face with improved error handling"""
#     try:
#         # Read image
#         img = cv2.imread(image_path)
#         if img is None:
#             raise Exception("Failed to load image")

#         # Convert to RGB and optimize size
#         height, width = img.shape[:2]
#         max_dimension = 800  # Limit maximum dimension
#         if max(height, width) > max_dimension:
#             scale = max_dimension / max(height, width)
#             img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(img_rgb)

#         # Detect face
#         face, prob = global_vars['mtcnn'](img_pil, return_prob=True)
#         if face is None:
#             return "No_Face_Detected"

#         # Get embedding
#         with torch.no_grad():
#             emb = global_vars['resnet'](face.unsqueeze(0))

#         # Find closest match
#         embedding_list, name_list = global_vars['saved_data']
#         distances = []
#         for idx, stored_emb in enumerate(embedding_list):
#             dist = torch.dist(emb.squeeze(), stored_emb.squeeze()).item()
#             distances.append(dist)
        
#         min_idx = distances.index(min(distances))
#         confidence = 1.0 / (1.0 + min(distances))  # Convert distance to confidence score
        
#         if confidence < 0.5:  # Threshold for minimum confidence
#             return "Unknown_Person"
            
#         return name_list[min_idx]

#     except Exception as e:
#         logger.error(f"Error in process_face: {str(e)}")
#         return f"Error_{str(e)}"




# def handler(event, context):
#     """Main handler with improved execution flow"""
#     try:
#         # Extract input parameters
#         if not isinstance(event, dict):
#             event = json.loads(event)
            
#         bucket_name = event.get('bucket_name')
#         image_file_name = event.get('image_file_name')
        
#         if not bucket_name or not image_file_name:
#             raise ValueError("Missing required parameters: bucket_name or image_file_name")
            
#         logger.info(f"Processing {image_file_name} from {bucket_name}")
        
#         # Initialize models
#         if not load_models():
#             raise Exception("Failed to initialize models")
        
#         # Local paths
#         local_image_path = f"/tmp/{image_file_name}"
        
#         # Download and process
#         if not download_if_not_exists(bucket_name, image_file_name, local_image_path):
#             raise Exception(f"Failed to download image from {bucket_name}/{image_file_name}")
        
#         # Process image
#         result_name = process_face(local_image_path)
        
#         # Prepare output
#         output_bucket = bucket_name.replace('-stage-1', '-output')
#         result_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        
#         # Upload result
#         s3.put_object(
#             Bucket=output_bucket,
#             Key=result_file_name,
#             Body=result_name
#         )
        
#         # Cleanup
#         if os.path.exists(local_image_path):
#             os.remove(local_image_path)
        
#         return {
#             'statusCode': 200,
#             'body': json.dumps({
#                 'message': f"Successfully processed {image_file_name}",
#                 'result': result_name,
#                 'output_location': f"{output_bucket}/{result_file_name}"
#             })
#         }

#     except Exception as e:
#         logger.error(f"Handler error: {str(e)}")
#         return {
#             'statusCode': 500,
#             'body': json.dumps({
#                 'error': str(e),
#                 'event': event
#             })
#         }

import boto3
import os
import logging
import json
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from botocore.exceptions import ClientError

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3 = boto3.client('s3')

# Global variables for model instances
global_vars = {
    'mtcnn': None,
    'resnet': None,
    'saved_data': None,
    'initialized': False
}

def download_if_not_exists(bucket, key, local_path):
    """Download file if it doesn't exist locally"""
    try:
        if not os.path.exists(local_path):
            logger.info(f"Downloading {key} from {bucket}")
            s3.download_file(bucket, key, local_path)
            logger.info(f"Download complete: {key}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading {key}: {str(e)}")
        return False

def load_models():
    """Initialize models with better error handling"""
    try:
        if global_vars['initialized']:
            return True

        logger.info("Starting model initialization")
        
        # Ensure we're using CPU
        torch.set_grad_enabled(False)
        
        # Download data.pt from your data bucket
        data_success = download_if_not_exists('1229421130-data', 'data.pt', '/tmp/data.pt')
        if not data_success:
            raise Exception("Failed to download data.pt")
            
        logger.info("Loading data.pt...")
        global_vars['saved_data'] = torch.load('/tmp/data.pt', map_location='cpu')
        
        logger.info("Initializing MTCNN...")
        global_vars['mtcnn'] = MTCNN(
            image_size=240,  # Changed to match handler2.py
            margin=0,
            min_face_size=20,
            device='cpu'
        )
        
        logger.info("Initializing ResNet...")
        global_vars['resnet'] = InceptionResnetV1(pretrained='vggface2').eval()
        
        global_vars['initialized'] = True
        logger.info("Model initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
        return False

def process_face(image_path):
    """Process face using handler2's recognition logic"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Failed to load image")

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Detect and get face
        face, prob = global_vars['mtcnn'](img_pil, return_prob=True)
        
        if face is None:
            return "No_Face_Detected"

        # Generate embedding
        emb = global_vars['resnet'](face.unsqueeze(0)).detach()

        # Get saved embeddings and names
        embedding_list, name_list = global_vars['saved_data']

        # Calculate distances
        dist_list = []
        for emb_db in embedding_list:
            dist = torch.dist(emb.squeeze(), emb_db.squeeze()).item()
            dist_list.append(dist)

        # Find closest match
        idx_min = dist_list.index(min(dist_list))
        
        # Add confidence threshold
        min_distance = min(dist_list)
        if min_distance > 1.0:  # You can adjust this threshold
            return "Unknown_Person"
            
        return name_list[idx_min]

    except Exception as e:
        logger.error(f"Error in process_face: {str(e)}")
        return f"Error_{str(e)}"

def handler(event, context):
    """Main handler with improved execution flow"""
    try:
        # Extract input parameters
        if not isinstance(event, dict):
            event = json.loads(event)
            
        bucket_name = event.get('bucket_name')
        image_file_name = event.get('image_file_name')
        
        if not bucket_name or not image_file_name:
            raise ValueError("Missing required parameters: bucket_name or image_file_name")
            
        logger.info(f"Processing {image_file_name} from {bucket_name}")
        
        # Initialize models
        if not load_models():
            raise Exception("Failed to initialize models")
        
        # Local paths
        local_image_path = f"/tmp/{image_file_name}"
        
        # Download and process
        if not download_if_not_exists(bucket_name, image_file_name, local_image_path):
            raise Exception(f"Failed to download image from {bucket_name}/{image_file_name}")
        
        # Process image
        result_name = process_face(local_image_path)
        
        # Prepare output
        output_bucket = bucket_name.replace('-stage-1', '-output')
        result_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        
        # Upload result
        s3.put_object(
            Bucket=output_bucket,
            Key=result_file_name,
            Body=result_name
        )
        
        # Cleanup
        if os.path.exists(local_image_path):
            os.remove(local_image_path)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f"Successfully processed {image_file_name}",
                'result': result_name,
                'output_location': f"{output_bucket}/{result_file_name}"
            })
        }

    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'event': event
            })
        }