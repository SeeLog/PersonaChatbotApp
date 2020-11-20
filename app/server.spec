# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

PATHEX = os.getcwd()

print("Path:", PATHEX)

a = Analysis(['persona_chatbot_app/bot/server.py'],
             pathex=[PATHEX],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)


#---------------追加ここから---------------
Key = ['mkl']
def remove_from_list(input, keys):
    outlist = []
    for item in input:
        name, _, _ = item
        flag = 0
        for key_word in keys:
            if name.find(key_word) > -1:
                flag = 1
        if flag != 1:
            outlist.append(item)
    return outlist
a.binaries = remove_from_list(a.binaries, Key)
#---------------追加ここまで---------------


exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='chatbot_server',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
