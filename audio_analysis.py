"""
------------Usage------------
CLI Commands

python audio_analysis.py -p "Path" -l "BCP-47 language tag"

argument -p : Path to audio files directory
argument -l : source audio language in BCP-47 language tag ('hi-IN', 'en-US', 'fr-FR').

default -l: 'hi-IN' 

(sript assumes that all audio files in provided path have source audio as hindi language.
should be chnaged to different language tap according to source aduio language.)

example : python audio_analysis.py -p C:/Users/sree3/Downloads/Music -l 'hi-IN'

----------------------------
First install required libraries

# pip install pyaudio
# pip install speechrecognition
# pip install googletrans
# pip install pydub
# pip install ffmpeg, ffprobe, simpleaudio
# pip install textblob
# pip install vaderSentiment

# Install ffmpeg source engine and add to path 
# (Used for conversion of different audio file formats into .wav audio file format.)
---------------------------

"""

# Import the libraries
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from googletrans import Translator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import sys
import shutil
from collections import Counter
import pandas as pd
from datetime import datetime as dt
import argparse
import re 
import pandas as pd
import json
import filetype
#from json_create import file_name


def get_metric_data(file_name):
    """
    Reads Metrics & Sub-Metrics from json file.
    """
    if os.path.exists(file_name):
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
                #print(data)
        except IOError:
            print('Error')
    return data

def audio_file(audio_file):
	"""
	Converting from different audio file formats to wav
	"""
	# Reading file name, file format for further uses
	#base = os.path.basename(audio_file)
	#file_name, file_format = os.path.splitext(base)

	file_name = audio_file.split('\\')[-1].split('.')[0]
	file_format = audio_file.split('.')[-1]
	#print('\naudio_file\n', file_name, '\n', file_format)
	if file_format == 'wav':
		# return the file name and path
		return  audio_file, file_name+'.wav'

	elif file_format != 'wav':
		try:
			# creating temp converted folder to store converted files
			folder_name = "converted"
			if not os.path.exists(os.path.join(path, folder_name)):
				os.mkdir(os.path.join(path, folder_name))
			
			conv_file_name = file_name + '.wav'
			# reading audio from path
			wav_audio = AudioSegment.from_file(audio_file , format=file_format)
			# exporting the converted file
			wav_audio.export(os.path.join(path, folder_name, conv_file_name), format="wav")
			# return the converted file name and path
			return os.path.join(path, folder_name, conv_file_name), conv_file_name
		except:
			print('Error occured while reading audio file...\nPlease use different file format (.wav, .mp3, .m4a are accepted)')


def get_large_audio_transcription(path, language):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # Create a speech recognizer instance
    r = sr.Recognizer()
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.listen(source)
            # try converting it to text
            try:
                # takes input language parameter 
                text = r.recognize_google(audio_listened, language=language)
            except sr.UnknownValueError as e:
                #print("Error:", str(e))
                pass
            else:
                text = f"{text.capitalize()}. "
                #print(chunk_filename, ":", text)
                #chunks_list.append(text)
                whole_text += text
    mydir= 'audio-chunks'
    ## Try to remove tree; if failed show an error using try...except on screen
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    # return the text for all chunks detected
    return whole_text


def translate(full_text):
    """
    Translating chunks from one language to other language and returns 
    a single string after joining converted chunks into one long text string.
    """
    translator = Translator()
    # breaking down text into small pieces for faster translation
    translations = []
    unique_elements = full_text.split('. ')

    for element in unique_elements:
            # Adding all the translations to a dictionary (translations)
            translations.append(translator.translate(element).text)
    # Final translated string
    english_translated_text = ''
    # joining all the converted chunks to final string
    for i in translations:
        english_translated_text = english_translated_text + ' ' + i
    # return the text for all translated chunks
    return english_translated_text


def remove_converted(path):
    """
    Deletes the temporary files created during sentiment analysis.
    """
    # Get directory name
    mydir= os.path.join(path, 'converted')
    # Try to remove tree; if failed show an error using try...except on screen
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        #print ("Error: %s - %s." % (e.filename, e.strerror))
        #print('Warning: This occurs only when there are no left over files from previous Audio Analysis.')
        #print('Warning: This does\'nt affect the current Audio Analysis.')
        pass


def vader_sentiment_analysis(sentence):
    """
    Perform sentiment analysis using vader framework
    and returns the sentiment.
    """
    # Instantiate analyser class and perform sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    # Obtain polarity scores
    vs = analyzer.polarity_scores(sentence)
    # Return the sentiment polarity scores
    return vs


def textblob_sentiment_analysis(sentence):
    """
    Perform sentiment analysis using textblob framework
    and returns the sentiment.
    """
    # Converting text to blob type and perform sentiment analysis
    blob = TextBlob(sentence)
    # Return the sentiment polarity scores
    return blob.sentiment


def audio_length(path):
    """
    Returns the duration/length of the audio file in seconds.
    """
    # Reading audio file from path
    song = AudioSegment.from_wav(path)
    # Obtaining duration in seconds with instance property
    time = song.duration_seconds
    
    # Scoring logic for duration of call
    if time < 30:
        score = 1
    elif time >= 30 and time <= 60:
        score = 2
    elif time > 60 and time <= 120:
        score = 3
    elif time > 120 and time <= 180:
        score = 3
    elif time > 180:
        score = 5

    # Return the duration of the audio file in seconds along with score.
    return str(time) + ' Sec', str(score)


def re_search(lst, text):
    """
    Returns list of all positive keyword matches from given
    list of keywords and text.
    """
    # Creating a return matches list object
    item_list = list()
    # Checking list items against the input text
    for item in lst:
        # Search method returns match if found, if item is not found returns None
        search_obj = re.search(item, text, re.IGNORECASE)
        if search_obj:
            # Appending positive matches to a list list_item
            item_list.append(search_obj.group())
    # Return the item list containing list of matched keywords in text
    return item_list


def call_type_score(specific_metric, specific_metric_keywords_match):
    """
    Returns a score for given metric and it's matched 
    keyword list based on predefined scoring criterion.
    """
    # 5 points metrics
    points_5 = ['COVID Profiling', 'IgG Positive Counselling', 'Immunity Boosting', 'IgG Negative Counselling', 
                'Appointment Booking', 'Appointment Reminder', 'TBCSS Intro']
    # 3 points metrics
    points_3 = ['Prescription clarification', 'Message Clarification', 'Test result / upload']
    # Length of matched keyword list
    match_length = len(specific_metric_keywords_match)
    # If input metric is not null and in 5 point list
    if specific_metric in points_5:
        if specific_metric_keywords_match and match_length >= 4:
            return 5
        elif specific_metric_keywords_match and match_length < 4:
            return match_length
    # If input metric is not null and in 3 point list
    if specific_metric in points_3:
        if specific_metric_keywords_match and match_length >= 4:
            return 3
        elif specific_metric_keywords_match and match_length < 4:
            return match_length     


def call_type(text, json_data):
    """
    Returns a dictionary with one or more metrics that
    defines what type of call it is.
    """
    # Creating data dictionary
    data = json_data
    # Creating a final list object
    call_type_curated = list()
    # Iterating over each metric in data dictionary
    for metric in list(data.keys()):
        # Obtaining keywords for a metric
        specific_metric_keywords = data[metric]
        # Obtaining keywords match list using re_search function
        specific_metric_keywords_match = re_search(specific_metric_keywords, text)
        # Obtaining score from call_type_score function
        score = call_type_score(metric, specific_metric_keywords_match)
        # Appending values to the final list
        call_type_curated.append((metric, specific_metric_keywords_match, len(specific_metric_keywords_match), score))
    # If no matches are found across all metrics return default score
    if not call_type_curated:
        return 'Other', 3
    # If matches are found across all metrics,
    if call_type_curated:   
        # Create a pandas data frame from the final list
        df = pd.DataFrame(call_type_curated, columns=['metric', 'match', 'length', 'score'])
        # Mask and select the metric with highest keyword matches
        df = df[df.length == max(df.length)]
        #print(df) 
        # Return dictionary with metric name & it's score
        return dict(zip(list(df['metric']), list(df['score'])))


def call_opening_or_closing_score(score, metric, specific_metric_keywords_match):
    """
    Returns a score for call opening and closure metrics.
    """
    # If metric is not negative and matches are not null
    if specific_metric_keywords_match and 'negative' not in metric:
        score += 1
        return score
    # If metric is negative and matches are not null
    elif specific_metric_keywords_match and 'negative' in metric:
        score -= 1
        return score
    # If matches are null
    elif not specific_metric_keywords_match:
        return score 


def call_opening(text, json_data):
    """
    Returns a score for call opening criterion.
    """
    # Creating data dictionary
    data = json_data
    # Initial score
    score = 0
    # Iterating over each metric in data dictionary
    for metric in list(data.keys()):
        # Obtaining keywords for a metric
        specific_metric_keywords = data[metric]
        # Obtaining keywords match list using re_search function
        specific_metric_keywords_match = re_search(specific_metric_keywords, text)
        # Obtaining score from call_opening_or_closing_score function
        score = call_opening_or_closing_score(score, metric, specific_metric_keywords_match)
    # Returns the final score value.
    return score


def patient_registration(text):
    """
    Returns list of all available phone numbers
    that are specified in the call.
    """
    return re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)


def call_handling_score(score, metric, specific_metric_keywords_match):
    """
    Returns score for given metric and it's matches
    """    
    match_length = len(specific_metric_keywords_match)
    # If metric is not negative and matches are not null
    if specific_metric_keywords_match and 'negative' not in metric:
        score += match_length
        return score
    # If metric is negative and matches are not null
    elif specific_metric_keywords_match and 'negative' in metric:
        score -= match_length
        return score
    # If matches are null
    elif not specific_metric_keywords_match:
        return score 


def call_handling(text, json_data):
    """
    Returns a score for call handling criterion
    """
    # Creating data dictionary
    data = json_data
    # Initial Score
    score = 0
    # Iterating over each metric in data dictionary
    for metric in list(data.keys()):
        # Obtaining keywords for a metric
        specific_metric_keywords = data[metric]
        # Obtaining keywords match list using re_search function
        specific_metric_keywords_match = re_search(specific_metric_keywords, text)
        # Obtaining score from call_handling_score function
        score = call_handling_score(score, metric, specific_metric_keywords_match)
    # If overall score is less than 0 return 1
    if score <= 0:
        return 1
    # If overall score is greater than 5 return 5
    elif score >= 5:
        return 5


def new_services(text, json_data):
    """
    Returns a score of 1 if any new services 
    are mentioned in call, else returs 0.
    """
    # New services keyword list.
    new_services = json_data
    # Obtaining keywords match list using re_search function
    service_keywords_match = re_search(new_services, text)      
    # If keyword matches are present 
    if service_keywords_match:
        return 1
    # If keyword matches are null or 0
    elif not service_keywords_match:
        return 0


def re_findall(lst, text):
    """
    Returns a list of all matching keywords
    from a given keyword list and text.
    """
    # Creating a return matches list object
    item_list = list()
    # Checking list items against the input text
    for item in lst:
        # Findall method returns all matches if found, if item is not found returns None
        search_obj = re.findall(item, text, re.IGNORECASE)
        if search_obj:
            # Appending multiple positive matches of same keyword to a list list_item
            item_list.extend(search_obj)
    # Return the item list containing list of matched keywords in text
    return item_list


def COVID_related_discussion(text, json_data):
    """
    Return a score based on frequency of word
    Covid used in the call.
    """
    # Covid keyword list
    covid_keywords = json_data
    # Obtaining all possible keywords matches list using re_findall function
    covid_keywords_match = re_findall(covid_keywords, text)     
    # If matches are not null
    if covid_keywords_match:
        # Find length of macthes
        covid_frequency = len(covid_keywords_match)
        # Scoring logic
        if covid_frequency >= 5:
            return 5
        elif covid_frequency == 4:
            return 4
        elif covid_frequency == 3:
            return 3
        elif covid_frequency == 2:
            return 2
        elif covid_frequency == 1:
            return 1
    # If matches are null
    elif not covid_keywords_match:
        return 0


def call_closure(text, json_data):
    """
    Returns a score for call closure criterion.
    """
    # Creating data dictionary
    data = json_data
    # Initial score
    score = 0
    # Iterating over each metric in data dictionary
    for metric in list(data.keys()):
        # Obtaining keywords for a metric
        specific_metric_keywords = data[metric]
        # Obtaining keywords match list using re_search function
        specific_metric_keywords_match = re_search(specific_metric_keywords, text)
        # Obtaining score from call_opening_or_closing_score function
        score = call_opening_or_closing_score(score, metric, specific_metric_keywords_match)
    # Return score value
    return score


def re_exception_search(lst, text, before_match=5, after_match=5):
    """
    Returns list of all positive keyword matches from given
    list of keywords and text.
    """
    # Creating a return matches list object
    item_list = list()
    # Creating a return sentences list object
    item_list_sentence = list()
    # Spliting the text into list for cross verification against search object span
    split_text = text.split(" ")
    # Checking list items against the input text
    for item in lst:
        # Search method returns match if found, if item is not found returns None
        search_obj = re.search(item, text, re.IGNORECASE)
        if search_obj:
            # Obtain start and end index of positive search match
            start, end = search_obj.span()
            # Split the text from start to end index
            split_keyword = text[start:end].split(" ")
            # If single match word
            if len(split_keyword) == 1:
                try:
                    # Obtain index from text
                    index = split_text.index(split_keyword[0])
                    if index:
                        # If index is returned, then it is a valid word
                        # Append the word to item_list
                        item_list.append(search_obj.group())
                        # Append the sentence where this matched keyword appeared
                        item_list_sentence.append(" ".join(split_text[index - before_match : index + after_match]))
                    else:
                        pass
                except:
                    pass
            # If match keyword is multiple words
            elif len(split_keyword) > 1:
                try:
                    # Obtain first and last word indexes
                    index_first = split_text.index(split_keyword[0])
                    index_last = split_text.index(split_keyword[-1])
                    # # If indexes is returned, then it is a valid word
                    if index_first and index_last:
                        # Append the word to item_list
                        item_list.append(search_obj.group())
                        # Append the sentence where this matched keyword appeared
                        item_list_sentence.append(" ".join(split_text[index - before_match : index + after_match]))
                    else:
                        pass
                except:
                    pass
    # Return the item_list containing list of matched keywords in text and 
    # item_list_sentence containing sentence where this matched keyword appeared
    return item_list, item_list_sentence


def exception_analysis(text, json_data, before_match, after_match):
    """
    Returns a score for each criterion 
    based on key wr=ord matching.
    """
    # Creating data dictionary
    data = json_data
    # Creating final match dictionary
    match_dict = dict()
    match_dict_keywords = dict()
    match_dict_descripton = dict()
    # Iterating over each metric in data dictionary
    for metric in list(data.keys()):
        # Obtaining keywords for a metric
        specific_metric_keywords = data[metric]
        # Obtaining keywords match list using re_search function
        specific_metric_keywords_match, sentences = re_exception_search(specific_metric_keywords, text, before_match, after_match)
        # Defining score as length of keyword matches for each metric
        match_dict[metric] = [len(specific_metric_keywords_match), specific_metric_keywords_match, sentences]
    # Return final metric and score dictionary
    return match_dict


def exception_analysis_sort_and_append_data(ea_data):
    """
    Seperates and appends combined  exception analysis 
    data dictionary into different category lists.
    """
    Patient_Status_list.append(ea_data['Patient Status'][0])
    Patient_Status_keywords_list.append(ea_data['Patient Status'][1])
    Patient_Status_description_list .append(ea_data['Patient Status'][2])
    Symptoms_list.append(ea_data['Symptoms'][0])
    Symptoms_keywords_list.append(ea_data['Symptoms'][1])
    Symptoms_description_list.append(ea_data['Symptoms'][2])
    Comorbidities_list.append(ea_data['Comorbidities'][0])
    Comorbidities_keywords_list.append(ea_data['Comorbidities'][1])
    Comorbidities_description_list.append(ea_data['Comorbidities'][2])
    Living_Condition_list.append(ea_data['Living Condition'][0])
    Living_Condition_keywords_list.append(ea_data['Living Condition'][1])
    Living_Condition_description_list.append(ea_data['Living Condition'][2])
    Living_with_list.append(ea_data['Living with'][0])
    Living_with_keywords_list.append(ea_data['Living with'][1])
    Living_with_description_list.append(ea_data['Living with'][2])
    Shared_Eating_list.append(ea_data['Shared Eating'][0])
    Shared_Eating_keywords_list.append(ea_data['Shared Eating'][1])
    Shared_Eating_description_list.append(ea_data['Shared Eating'][2])
    Private_Eating_list.append(ea_data['Private Eating'][0])
    Private_Eating_keywords_list.append(ea_data['Private Eating'][1])
    Private_Eating_description_list.append(ea_data['Private Eating'][2])
    Shared_Transport_list.append(ea_data['Shared Transport'][0])
    Shared_Transport_keywords_list.append(ea_data['Shared Transport'][1])
    Shared_Transport_description_list.append(ea_data['Shared Transport'][2])
    Self_Transport_list.append(ea_data['Self Transport'][0])
    Self_Transport_keywords_list.append(ea_data['Self Transport'][1])
    Self_Transport_description_list.append(ea_data['Self Transport'][2])
    Working_as_list.append(ea_data['Working as'][0])
    Working_as_keywords_list.append(ea_data['Working as'][1])
    Working_as_description_list.append(ea_data['Working as'][2])
    Number_of_work_contacts_list.append(ea_data['Number of work contacts'][0])
    Number_of_work_contacts_keywords_list.append(ea_data['Number of work contacts'][1])
    Number_of_work_contacts_description_list.append(ea_data['Number of work contacts'][2])

def vader_sentiment_sort(vader_sentiment):
    """
    Seperates and appends vader sentiment scores 
    dictionary into different category lists.
    """
    vader_neg.append(vader_sentiment['neg'])
    vader_neu.append(vader_sentiment['neu'])
    vader_pos.append(vader_sentiment['pos'])
    vader_compound.append(vader_sentiment['compound'])


def text_blob_sentiment_sort(text_blob_sentiment):
    """
    Seperates and appends textblob sentiement scores
    tuple into different category lists.
    """
    text_blob_polarity.append(text_blob_sentiment[0])
    text_blob_subjectivity.append(text_blob_sentiment[1])


def file_type(path):
    """
    Returns a list of valid audio file names
    from the given path with one or more file types.
    """
    audio_files = list()
    # Searching in given path
    for root, _, files in os.walk(path, topdown=False):
        # For each file in list of files
        for filename in files:
            try:            
                kind = filetype.guess(os.path.join(root, filename))
                # If file type is audio append to audio_file list
                if kind.mime.split('/')[0] == 'audio':
                    audio_files.append(filename)
            except:
                # If file type is not determined then pass
                pass
    # Return list of audio files for given path
    return audio_files


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# Adding argument to parser
# Parameter path 
ap.add_argument("-p", "--path", required=True,
    help="path to directory of audio files/ file.")
# Parameter source audio language 
ap.add_argument("-l", "--language", required=False, default='hi-IN',
    help="source audio language in BCP-47 language tag ('hi-IN', 'en-US', 'fr-FR').")

args = vars(ap.parse_args())

# Retrieve path from args to path variable
path = args["path"]
language = args["language"]
# Remove previous converted files
remove_converted(path)  

# List all the audio files in the target directory
#files = os.listdir(path)
#files = [file for file in files if file.split(".")[-1] == 'wav']
files = file_type(path)
print("\nAudio Files :", files)

# Creating temp data lists
file_name_list = list()
file_path_list = list()
original_audio_transcription_list = list()
english_translated_text_list = list()
# Bridgital Effectiveness Metrics
audio_duration_list = list()
audio_score_list = list()
call_type_score_list = list()
call_opening_score_list = list()
patient_registration_numbers_list = list()
call_handling_score_list = list()
new_services_score_list = list()
COVID_related_discussion_frequency_list = list()
call_closure_score_list = list()
vader_neg, vader_neu, vader_pos, vader_compound = list(), list(), list(), list()
text_blob_polarity, text_blob_subjectivity = list(), list()
# Exception Analysis
Patient_Status_list, Patient_Status_keywords_list, Patient_Status_description_list = list(), list(), list()
Symptoms_list, Symptoms_keywords_list, Symptoms_description_list = list(), list(), list()
Comorbidities_list, Comorbidities_keywords_list, Comorbidities_description_list = list(), list(), list()
Living_Condition_list, Living_Condition_keywords_list, Living_Condition_description_list = list(), list(), list()
Living_with_list, Living_with_keywords_list, Living_with_description_list = list(), list(), list()
Shared_Eating_list, Shared_Eating_keywords_list, Shared_Eating_description_list = list(), list(), list()
Private_Eating_list, Private_Eating_keywords_list, Private_Eating_description_list = list(), list(), list()
Shared_Transport_list, Shared_Transport_keywords_list, Shared_Transport_description_list = list(), list(), list()
Self_Transport_list, Self_Transport_keywords_list, Self_Transport_description_list = list(), list(), list()
Working_as_list, Working_as_keywords_list, Working_as_description_list = list(), list(), list()
Number_of_work_contacts_list, Number_of_work_contacts_keywords_list, Number_of_work_contacts_description_list = list(), list(), list()

print('\nStarting Audio Analysis')

# Importing data from existing json file
file_name = 'voice_call_analysis_data.json'
data = get_metric_data(file_name)

# Itterating over all files in target directory
# For each file obtain the result and 
# Append results to their respective final list
for file in files:
    print('...')
    # Read Audio File
    audio_wav, file_name = audio_file(os.path.join(path, file))
    file_name_list.append(file_name)
    file_path_list.append(audio_wav)
    
    # Convert audio to text
    original_audio_transcription = get_large_audio_transcription(audio_wav, language)
    original_audio_transcription_list.append(original_audio_transcription)

    # Convert text to english text
    english_translated_text = translate(original_audio_transcription)
    english_translated_text_list.append(english_translated_text)

    # Get length/duration of the audio file
    audio_duration, audio_score = audio_length(audio_wav)
    audio_duration_list.append(audio_duration)
    audio_score_list.append(audio_score)

    # Obtain Call type from text
    call_type_score_dict = call_type(english_translated_text, data['Call Type'][0])
    call_type_score_list.append(call_type_score_dict)

    # Obtain Call opening score from text
    call_opening_score = call_opening(english_translated_text, data['Call Opening'][0])
    call_opening_score_list.append(call_opening_score)

    # Obtain Patient registration numbers from text
    patient_registration_numbers = patient_registration(english_translated_text)
    patient_registration_numbers_list.append(patient_registration_numbers)

    # Obtain Call handling Score from text
    call_handling_value = call_handling(english_translated_text, data['Call Handling'][0])
    call_handling_score_list.append(call_handling_value)

    # Obtain New Services score from text
    new_services_score = new_services(english_translated_text, data['New Services'][0]['New Services'])
    new_services_score_list.append(new_services_score)

    # Obtain COVID related discussion frequency score from text
    COVID_related_discussion_frequency = COVID_related_discussion(english_translated_text, data['Covid Keywords'][0]['Covid Keywords'])
    COVID_related_discussion_frequency_list.append(COVID_related_discussion_frequency)

    # Obtain Call closure score from text
    call_closure_score = call_closure(english_translated_text, data['Call Closure'][0])
    call_closure_score_list.append(call_closure_score)

    # Predicts the sentiment of the text using Vader Sentiment Analysis
    vader_sentiment = vader_sentiment_analysis(english_translated_text)
    vader_sentiment_sort(vader_sentiment)

    # Predicts the sentiment of the text using Text blob
    text_blob_sentiment = textblob_sentiment_analysis(english_translated_text)
    text_blob_sentiment_sort(text_blob_sentiment)

    # Obtain Exception Analysis scores for each criterion
    exception_analysis_dict = exception_analysis(english_translated_text, data['Exception Analysis'][0], 5, 5)
    exception_analysis_sort_and_append_data(exception_analysis_dict)

    print(file_name, 'file analysed.')

# Creating data frame with all final lists 
data = pd.DataFrame({'File Name': file_name_list, 'File Path': file_path_list, 
        'Original Audio Transcription': original_audio_transcription_list, 'English Translated Text': english_translated_text_list, 
        'Audio Duration': audio_duration_list, 'Audio Score': audio_score_list, 'Call Type': call_type_score_list, 
        'Call Opening': call_opening_score_list, 'Patient Registration Number': patient_registration_numbers_list, 
        'Call Handling': call_handling_score_list, 'New Services': new_services_score_list,
        'COVID Related Discussion': COVID_related_discussion_frequency_list, 'Call Closure': call_closure_score_list, 
        'Vader Negative Sentiment': vader_neg, 'Vader Neutral Sentiment': vader_neu, 
        'Vader Positive Sentiment':vader_pos, 'Vader Compound Sentiment':vader_compound,
        'Text Blob Polarity': text_blob_polarity, 'Text Blob Subjectivity': text_blob_subjectivity,
        'Patient_Status': Patient_Status_list, 'Patient_Status_keywords': Patient_Status_keywords_list, 'Patient_Status_description': Patient_Status_description_list,
        'Symptoms': Symptoms_list, 'Symptoms_keywords': Symptoms_keywords_list, 'Symptoms_description': Symptoms_description_list,
        'Comorbidities': Comorbidities_list, 'Comorbidities_keywords': Comorbidities_keywords_list, 'Comorbidities_description': Comorbidities_description_list,
        'Living_Condition': Living_Condition_list, 'Living_Condition_keywords': Living_Condition_keywords_list, 'Living_Condition_description': Living_Condition_description_list,
        'Living_with': Living_with_list, 'Living_with_keywords': Living_with_keywords_list, 'Living_with_description': Living_with_description_list,
        'Shared_Eating': Shared_Eating_list, 'Shared_Eating_keywords': Shared_Eating_keywords_list, 'Shared_Eating_description': Shared_Eating_description_list,
        'Private_Eating': Private_Eating_list, 'Private_Eating_keywords': Private_Eating_keywords_list, 'Private_Eating_description': Private_Eating_description_list,
        'Shared_Transport': Shared_Transport_list, 'Shared_Transport_keywords': Shared_Transport_keywords_list, 'Shared_Transport_description': Shared_Transport_description_list,
        'Self_Transport': Self_Transport_list, 'Self_Transport_keywords': Self_Transport_keywords_list, 'Self_Transport_description': Self_Transport_description_list,
        'Working_as': Working_as_list, 'Working_as_keywords': Working_as_keywords_list, 'Working_as_description': Working_as_description_list,
        'Number_of_work_contacts': Number_of_work_contacts_list, 'Number_of_work_contacts_keywords': Number_of_work_contacts_keywords_list, 
        'Number_of_work_contacts_description': Number_of_work_contacts_description_list })

# Adding date to the name of .csv file
date = str(dt.today())
date = date.split('.')[0].replace(' ', '_').replace(':', '-')
# Converting dataframe into .csv file
saving_csv_file_name = path +r'\audio_analysis_' + date + '.csv'
data.to_csv(saving_csv_file_name, index=False)
# Delete the converted files for better memory management
remove_converted(path)
# Display end message
print(f'\nAudio Analysis complete. Output file is saved at {saving_csv_file_name}')