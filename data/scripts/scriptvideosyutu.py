import yt_dlp

url = "https://www.youtube.com/watch?v=LeYIndII13w"

ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': 'video_2.mp4',
    'merge_output_format': 'mp4',
    'download_ranges': yt_dlp.utils.download_range_func(None, [(0, 300)])
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])