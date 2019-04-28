import src.musicLoading as music

if __name__ == '__main__':
    music1 = music.get_all_midis_in_folder("Music/kunstderfuge.com")
    music1 += music.get_all_midis_in_folder("Music/piano-midi.de")


    for song in music1:
        out_path = song[1]
        out_path = "Quantized Music" + out_path[5:]
        music.quantized_to_midi(song[0], out_path)

