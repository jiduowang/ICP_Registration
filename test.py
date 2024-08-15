from CSF2 import CSF2

if __name__ == '__main__':
    csf1 = CSF2('./data/066.ply', './data/066_ground.ply', filetype='ply')
    csf1.process()
    csf2 = CSF2('./data/079.ply', './data/079_ground.ply', filetype='ply')
    csf2.process()
