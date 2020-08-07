#0. access server
```
sudo ssh -i google_compute_engine.dms dev@34.105.49.114
cd ~/thang/detectron2/
```
#1. install google cloud and login
https://cloud.google.com/sdk/docs
```
gcloud auth login
```

#2. create image
```
export PROJECT_ID=ai-project-231602
export JOB_DIR=gs://hptuning_scratch
export IMAGE_REPO_NAME=scartch_tuning_container
export IMAGE_TAG=scratch_tuning
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
export REGION=us-west1
export JOB_NAME=hp_tuning_scratch_container_job_$(date +%Y%m%d_%H%M%S)

docker build --no-cache --build-arg USER_ID=$UID -f Dockerfile -t $IMAGE_URI ./
```

#3. test docker (optional)
```
docker run $IMAGE_URI
```

#4. find ID images
```
docker images
```
#5.push images to google cloud (may encounter bug but keep pushing until it done)
// replace IMAGES_ID  which have tag "scratch_tuning" to images_ID
```
docker tag images_ID $IMAGE_URI
docker push $IMAGE_URI
```
#5.tuning
```
cd prscratch

export JOB_NAME=hp_tuning_container_job_$(date +%Y%m%d_%H%M%S)
gcloud beta ml-engine jobs submit training $JOB_NAME \
  --job-dir=$JOB_DIR \
  --region=$REGION \
  --master-image-uri $IMAGE_URI \
  --config=hyper_tuning.yaml

```

#6. check model
```
gsutil ls gs://hptuning/*
```

#7. download model
```
gsutil cp gs://hptuning/path/to/model
```
#8. remove images
```
sudo docker rmi -f images_ID
```
