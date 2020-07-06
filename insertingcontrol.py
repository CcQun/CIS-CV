# -*- coding: utf-8 -*-
'''
将事件插入数据库的控制程序-控制 seconds 秒可插入1次

用法：

'''

import time

from oldcare.utils.pathassistant import get_path

seconds = 1  # 每经过60秒才允许再次插入

insert_control_file_path = get_path('insert_control_file_path')

while True:
    f = open(insert_control_file_path, 'r')
    content = f.read()
    f.close()

    allow = content[11:12]

    if allow == '0':
        print('status: not allow')
        for i in range(seconds, 0, -1):
            print('wait %d seconds...' % (i))
            time.sleep(1)

        f = open(insert_control_file_path, 'w')
        f.write('is_allowed=1')
        f.close()

    elif allow == '1':
        print('status: allow')
        time.sleep(1)
    else:
        pass