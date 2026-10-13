import numpy as np
from sentence_transformers import SentenceTransformer
#making variable for filename
filename="WW2.txt" #WW2.txt was the file name i used for testing- the wikipedia page(only text) about world war 2 
with open(filename,"r") as f:
    content=f.read()

chunk=""
chunks=[]

sentences=[]
current=""
totalchlen=0
senlen=0
for char in content:
        current+=char
        if char == "." or char=="?" or char=="!":
            sentences.append(current) 
            current=""
for sentence in sentences:
        senlen=len(sentence.split())
        totalchlen+=senlen
        chunk+=" "+sentence
        if totalchlen>=40:
            chunks.append(chunk)
            chunk=sentence #for overlap
            totalchlen=senlen
if chunk!="":
        chunks.append(chunk)
             
#using sentence transformers library to encode 
model=SentenceTransformer("all-MiniLM-L6-v2")
embeddedchunks=model.encode(chunks)
chunkno=len(embeddedchunks)
#taking query from user then encoding it
query=input("Enter your query: ")
queryembed=model.encode(query)
pairs={} #to store each chunk's score
queryembedmag=np.linalg.norm(queryembed)
for i in range(0,chunkno):
      embeddedchunksmag=np.linalg.norm(embeddedchunks[i])
      score=((np.dot(queryembed,embeddedchunks[i]))/(queryembedmag*embeddedchunksmag))
      pairs[chunks[i]]=score 
#now to find top 5 scores 
topsearches=[]
topscores=[]
scorelist=list(pairs.values())
upper=len(scorelist)
chunklist=list(pairs.keys())
threshold=0.471
for count in range(0,min(5,upper)):  
    highscore=max(scorelist)
    index=scorelist.index(highscore)
    #removing the max value of this iteration from the list
    scorelist.pop(index)
    highchunk=chunklist.pop(index)
    if highscore>=threshold:
        topsearches.append(highchunk)
        topscores.append(highscore)
if (len(topscores))>0:
    print("Here are the top search results for your query")
    for i in range(0,min(5,len(topscores))):
        print("Search Result Number ",(i+1),": ",topsearches[i],"\n\n\n")
else:
     print("No relevant search results found")
      


