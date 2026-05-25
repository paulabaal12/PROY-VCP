import yt_dlp

url = "https://www.youtube.com/watch?v=bkWa_UAslE0"

#https://www.youtube.com/watch?v=LeYIndII13w

ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': 'video_cotorro.mp4',
    'merge_output_format': 'mp4',
    'download_ranges': yt_dlp.utils.download_range_func(None, [(0, 300)]),
    'cookiefile': 'www.youtube.com_cookies.txt',  
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])