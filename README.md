# Reinforcement-Learning-2d-Sailing
Using reinforcement learning to enable an agent to sail around the course



## Overview
As a runner it is always important to focus on and improve running form in order to increase your performance and to prevent injury. One of the most beneficial things you can do is run with a fast cadence. Not only does this lead to a faster pace but also the increased cadence causes your feet to land underneath you. This means you land on your fore foot rather than mid foot which is very beneficial for your muscles and bones. I always struggle to keep track of my cadence, and even with a running watch it's not fun to keep looking at it all the time!

**To solve this issue here is the project!!**

Since I always run with music it is the perfect tool to act as a cadence guide, whilst also not being as boring as a metronome! This script initially creates recommendations of songs from a list of playlists you set. Then using your top 50 current playlists, with the user rating "1" any playlists that they use for running (or would find good to run to), it predicts the songs out of the recommended songs that would suit your running playlist best. It then imports your garmin data (sorry if you don't have a garmin device!), and extracts your average running cadence from runs over a set min/km. If you haven't got a Garmin device the script can be easily manually inputted with a cadence range that you decide, or is calculcated from another source eg strava, whoop, etc. This Steps Per Minute (SPM) range is then fed into the playlist creation script and cuts any songs that aren't within the same BPM range. If you want to train for a faster cadence I have added an offset value to increase the uppper SPM limit for faster songs. After that out comes a playlist of whole new running songs that should inspire you to run, as well as act as the **perfect cadence training tool!!**

The notebook compares 4 common ML models and you can select which one is the best at predicting your music, or even tune some of the parameters. Then adjust the playlist creation script with this model for optimal results.

Please let me know if this helps anyone improve their running and I am always open to suggestions for improving these scripts!

## Reward Timeline
initailly just points for through gate
then wanted to add dynamic so every time closer is point and further is a point - it started to just crash into wall to die early since points not balanced well, ie not enough negative for hitting wall
then added time dependancy to both

## Usage
1) Set up your Spotify API.
2) Change spotify_credentials.py to work with your credentials.
3) Adjust the username of the cache file on music_data.py, without indicating cache location spotipy often timed out for me (the normal token limit is 1hr unless it refreshes).
4) Also in music_data.py adjust which playlists the script will create recommendations from, this process takes a while to run! So I limited mine to a small amount, but if you're happy to wait do as many as you please.
5) Run music_data.py which will create two pickle files of the recommended songs and your playlist songs.
6) Download your data from garmin connect and place within the source folder.
7) Work through garmin_data_analysis.ipynb and check your values for Upper and Lower Cadence (SPM), adjust the offsetter if required.
8) Within model_comparison.ipynb adjust the names of your favourite playlists, these are the playlists that you will "rate 1" to train the model, I used playlists that I have run to before, or any playlist of upbeat songs that I can imagine would uplift my run.
9) Work through model_comparison.ipynb to decide which model works best with your data.
10) Adjust the code in playlist_creation.py for a different model, playlist name, etc if required.
11) Run playlist_creation.py and enjoy the results! Ideally it should help you achieve faster running times, or at least better running form!!!

## Requirements to run:
* Spotify account
* Garmin Connect account
* Install the required libraries below

## Libraries:
* pandas
* spotipy
* imblearn
* sklearn
* XGBoost

## Next Steps:
* Inverse RL
* Enable better speed ie like it doesnt just stop when turning
* maybe add foiling so like needs to keep speed up
* dynamic wind direction
* 
