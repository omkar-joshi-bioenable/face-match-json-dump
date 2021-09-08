from datetime import datetime,timedelta
import base64
import time
import numpy as np
import os
from annoy import AnnoyIndex
from keras.models import load_model
from google.cloud import storage 
import json
import time
import requests
import cv2

import pathlib
import time
from scipy.spatial.distance import cosine
import dlib

from fastapi import Request, FastAPI
import uvicorn
from starlette.responses import Response
from collections import Counter ,defaultdict
#tf.get_logger().setLevel('ERROR')

storage_client=storage.Client()

main_dir = str(pathlib.Path(__file__).absolute().parent)

#load models
shape_predictor = os.path.join(main_dir,'model_files','shape_predictor_5_face_landmarks.dat')
dlib_face_recognition_model = os.path.join(main_dir,'model_files','dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor)
facerec = dlib.face_recognition_model_v1(dlib_face_recognition_model)

'''
# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	face_pixels = cv2.resize(face_pixels, (160,160), interpolation = cv2.INTER_AREA)
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def download_facenet_model(model_path):    
    if not os.path.isfile(model_path):
        model_url = 'https://storage.googleapis.com/bio-colab/face/facenet_keras.h5'
        print('Downloading model......')
        r = requests.get(model_url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    else:
        print('Model already downloaded')

 '''       
def download_load_annoy_json(temp_dir,model_bucket):
  try:            
      local_ann_index_file_path = os.path.join(temp_dir, os.path.basename(cloud_ann_index_file_path))
      download_blob(model_bucket,cloud_ann_index_file_path,local_ann_index_file_path)
      print('Model downloaded')                        
      sequence_in_index_json_file_local_path = os.path.join(temp_dir,os.path.basename(sequence_in_index_json_file))
      download_blob(model_bucket, sequence_in_index_json_file, sequence_in_index_json_file_local_path)
      print('json file downloaded')
      print('Loading annoy model')
      u = AnnoyIndex(128, 'angular')
      u.load(local_ann_index_file_path)
      print('Reading sequence json file')
      with open(sequence_in_index_json_file_local_path) as json_file:
        filenames = json.load(json_file)
      return u,filenames
    
  except Exception as e:
    print('Exception in download_load_annoy_json : ',e)
    return None,None



def image_search(features):
  start = time.time()
  try:
    if len(features) != 0:  
        # extracting features
        print('extracting embedding features')
        #features = get_embedding(model, image_array)
        print('evaluating output')
        # evaluating output
        indices, distances = u.get_nns_by_vector(features, 5, include_distances=True)
        print("Indices :",indices)
        print("Distances :",distances)
        num_features = len(distances)
        print('filenames : ',filenames)
        if len(distances) > 0 :
            name = filenames[indices[0]].split('/')[-3]
            name2 = filenames[indices[0]]
            distance = distances[0]
            end = time.time()
            time_taken = end - start
            print("Complete Name :",name2)
            print(name,distance)
            return (name,distance)
        else:
            print('Not found')
            return 0
  except Exception as e:
    print('Exception in image_search : ',e)


def load_settings():
  #sheet_url="https://script.google.com/a/bioenabletech.com/macros/s/AKfycbz6mnBNf__46yiIo6WEA0ljkxRBLz5nrQ5ccLQcAA/exec?"
  #data=requests.get(sheet_url+"page=matching").json()
  #print("settings_data=",data)
  image_save_bucket='c_function_test_bucket2'
  request_save_bucket='c_function_test_bucket3'
  default_folder='face_registration'
  default_organization='organization_1'
  default_project='project_1'
  c_function_url="https://asia-south1-cityairapp.cloudfunctions.net/face-match3"
  default_user_id="user_id"
  storage_path_name='storage_path'
  organization_name='organization'
  project_name='project'
  transaction_id_name='transaction_id'
  user_id_name='user_id'
  sleep_time=0
  return image_save_bucket,request_save_bucket,default_folder,default_organization,default_project,c_function_url,default_user_id,storage_path_name,organization_name,project_name,transaction_id_name,user_id_name,sleep_time


def upload_json(bucket, destination_jsonfile_name, result_json):
  bucket.blob(destination_jsonfile_name).upload_from_string(data=json.dumps(result_json),content_type='application/json')
  print('Data uploaded to {}.'.format(destination_jsonfile_name))

def download_blob(bucket, source_blob_name, destination_file_name):
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))


cloud_ann_index_file_path = 'face_registration/organization_1/project_1/face_registration-organization_1-project_1-index.ann'
sequence_in_index_json_file = 'face_registration/organization_1/project_1/face_registration-organization_1-project_1-file_sequence_in_index.json'

temp_dir = '/tmp'
#model_path = os.path.join(temp_dir,'facenet_keras.h5')



# download facenet model
print('global -- download facenet model')
#download_facenet_model(model_path)
# load facenet model
#model = load_model(model_path)

# download and load annoy model

app = FastAPI()

@app.post("/")
async def face_matching(request: Request):
  try:
    start=time.time()
    payload = await request.json()
    if not payload:
        msg = "no message received"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    # "data" will be present in normal format, "message" will be present in pubsub format       
    if "data" in payload or "message"in payload:
        if "data" in payload:
            payload = payload["data"]
            image_bucket = payload['bucket']
            imgfile_path = payload['name']
        if "message"in payload:
            payload = base64.b64decode(payload["message"]["data"]).decode("utf-8").strip()
            payload = json.loads(payload)
            image_bucket = payload['bucket']
            imgfile_path = payload['name']

    timestamp=str(datetime.now()).replace("-","").replace(":","").replace(".","-").replace(" ","T")
    print('Images of supported extensions allowed')
    supported_extensions = ['.jpg','.jpeg','.png','.JPG','.JPEG','.PNG','.jfif','.Jpeg'] 
    print('Data : filename and bucket name of image uploaded')

    file_name_with_path = imgfile_path
    print('file_name_with_path ',file_name_with_path)
    file_name = os.path.basename(file_name_with_path)
    print('file_name ',file_name)
    dir_path = file_name_with_path.split(file_name)[0]
    print('dir_path ',dir_path)
    bucket_name = image_bucket
    input_bucket = storage_client.get_bucket(bucket_name)
    output_bucket = storage_client.get_bucket("fs_match_json_resp_dump_bckt")

    

    global cloud_ann_index_file_path,sequence_in_index_json_file
    c_path_main_folder=file_name_with_path.split("/")[0]
    c_path_user_id=file_name_with_path.split("/")[2]
    cloud_ann_index_file_path=os.path.join("face_registration",c_path_main_folder,c_path_user_id,"index.ann")
    sequence_in_index_json_file=os.path.join("face_registration",c_path_main_folder,c_path_user_id,"file_sequence_in_index.json")
    print("cloud_ann_index_file_path =",cloud_ann_index_file_path)
    print("sequence_in_index_json_file =",sequence_in_index_json_file)
    global u,filenames
    ann_bucket=storage_client.get_bucket("fs_registr_one2one_map")
    u,filenames = download_load_annoy_json(temp_dir,ann_bucket)
    extension = pathlib.Path(file_name).suffix
    if extension in supported_extensions:            
        print('Downloading image....')
        download_image_path = os.path.join(temp_dir,file_name)
        download_blob(input_bucket, file_name_with_path,download_image_path)

        print('Image file present-confirmed ', os.path.isfile(download_image_path))
       
        # extracting face from image and saving it for further operations
        #image_array = cv2.imread(download_image_path)
        image_array=dlib.load_rgb_image(download_image_path)	
        try:
          image_array_time=time.time()
          print("Time taken for getting image_array=",image_array_time-start)
          try:
            img1_detection = detector(image_array, 1)
            if len(img1_detection)==1:		
                img1_shape = sp(image_array, img1_detection[0])
                img1_aligned = dlib.get_face_chip(image_array, img1_shape)            
                img1_representation = facerec.compute_face_descriptor(img1_aligned)            
                img1_representation = np.array(img1_representation)
                name,distance = image_search(img1_representation)
                save_json_data = {'status':'success','transaction_id':timestamp,'datetime':str(datetime.now()),'distance': distance,'InputFilePath':file_name_with_path,"InputBucket":bucket_name,"message":"","face_result":[[img1_detection[0].left(),img1_detection[0].top()],[img1_detection[0].right(),img1_detection[0].bottom()]]}
                destination_jsonfile_name=file_name_with_path.split(".")[0]+".json"
                cv2.imwrite("cropped_image.jpg",image_array[img1_detection[0].top():img1_detection[0].bottom(),img1_detection[0].left():img1_detection[0].right()])				
                output_bucket.blob(destination_jsonfile_name.repalce("json","jpg")).upload_from_filename("cropped_image.jpg")
		upload_json(output_bucket, destination_jsonfile_name, save_json_data)
                return ("Run successfully", 204)                
            else:
                result = [ [i.left(),i.top(),i.right(),i.bottom()] for i in img1_detection]
                save_json_data = {'status':'failure','transaction_id':timestamp,'datetime':str(datetime.now()),'distance': "Null",'InputFilePath':file_name_with_path,"InputBucket":bucket_name,"message":"invalied number of faces : "+str(len(img1_detection)),"face_result":result}
                destination_jsonfile_name=file_name_with_path.split(".")[0]+".json"
                upload_json(output_bucket, destination_jsonfile_name, save_json_data)
                return ("Run successfully", 204)
          except Exception as e:
            print("Exception occured in mtcnn face detection",e)
            return ("Exception occured in mtcnn face detection", str(e))
        except Exception as e:
          print("Exception occured after handle_request",e)
          return ("Exception occured after handle_request", str(e))
      
  except Exception as e:
    print("Exception in face_matching function",e)
    return ("Exception in face_matching function", str(e))

            
if __name__ == '__main__':
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8000
    uvicorn.run(app,host='0.0.0.0',port=8000)    

