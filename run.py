import os
"""
with open('commit_number.txt', 'r+') as f:
    commit_number = int(f.read())
    f.seek(0)
    f.write(str(commit_number + 1))
    f.truncate()
"""

commit_number = 1
os.system('git add .')
commit_message = 'Ex1 - ' + str(commit_number) + '. commit'
os.system('git commit -a -m "' + commit_message + '"')
os.system('git push')

