# RAG AI Teaching Assistant

This is an RAG based AI chatbot that trains on the given videos and answers the questions by refering the exact timestamp from the videos

#Problem Statement

Sometimes when we are in hurry or need just one topic from the pile of videos it is very time consuming and frustrating, Here comes the RAG AI teaching Assistant.

Lets take a scenario,
Tomorrow you have exam and you are revising from your notes and you cannot understand a specific topic and you go to your video tutorials, it would take time to find your required topic and it will result in wastage of time and energy.
With this Ai assistant you can feed your videos to it and just ask about the topic you want and wholla! you get the exact timestamps and the exact video number along with the summary.
This does not only works for tutorial but with any kind of videos

# How to use this RAG AI Teachiong Assistant

## Step:1 - Collect your videos
Move all your video files to the video folder 

## Step:2 - Convert videos to mp3
Convert all the videos to mp3 by running 'video_to_mp3.py' file
- It uses ffmpeg to convert videos to mp3

## Step:3 - Convert mp3 to json chunks
Convert all the mp3s to json chunks by running 'mp3_to_json.py' file
- It uses Whisper to convert mp3s to text chunks and then storing them to json files
- you can modify the model of whisper according to your need and the specs of your pc

## Step:4 Convert the json files to Vectors
Use the file 'preprocess_json.py' to convert json chunks into Vector Embeddings and save them as a joblib pickle.
- Instead of joblib pickle you can also use Vector Database like chroma and FSSAI

## Step:5 Prompt generation and feeding to LLM
Read the joblib file  and load it into the memory.
Use the file 'process_incoming.py' for this step


