import examples
from make_music import results_to_midi

def load_then_save(pathIn, pathout):
    midData = examples.load_data(pathIn)
    notes = examples.remove_rhythm(midData)
    results_to_midi(notes, pathout)

    midData2 = examples.load_data(pathout)
    notes2 = examples.remove_rhythm(midData2)

    results_to_midi(notes2, pathout + '2.mid')

    midData3 = examples.load_data(pathout + '2.mid')
    notes3 = examples.remove_rhythm(midData3)

    results_to_midi(notes3, pathout + '3.mid')

    if len(notes3) != len(notes2):
        print(pathout, 'no match overall length')
        return False

    for i in range(len(notes2)):
        if len(notes2[i]) != len(notes3[i]):
            print(pathout, "no match chord", i)
            return False
        for k in range(len(notes2[i])):
            if notes2[i][k] != notes3[i][k]:
                print(pathout, "no match note in pos ", i, k)
                return False

    print(pathout, "matches!")
    return True

if __name__ == '__main__':
    load_then_save('/home/whomagoo/github/MLMusic/Music/kunstderfuge.com/scarlatti 259.mid', '256.mid')
    load_then_save('/home/whomagoo/github/MLMusic/Music/kunstderfuge.com/scarlatti 187.mid', '187.mid')
    load_then_save('/home/whomagoo/github/MLMusic/Music/kunstderfuge.com/scarlatti 65.mid', '65.mid')
    load_then_save('/home/whomagoo/github/MLMusic/Music/kunstderfuge.com/scarlatti 47.mid', '47.mid')
    load_then_save('/home/whomagoo/github/MLMusic/Music/kunstderfuge.com/scarlatti 108.mid', '108.mid')
    load_then_save('/home/whomagoo/github/MLMusic/Music/kunstderfuge.com/scarlatti 162.mid', '162.mid')

    print("finished")
