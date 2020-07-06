# -*- coding: utf-8 -*-
'''
将事件插入数据库主程序

用法：

'''

import datetime
import json
import requests


def inserting(event_desc, event_type, event_location, old_people_id):
    url = "http://localhost:60000/eventInfo/addEvent"

    f = open('allowinsertdatabase.txt', 'r')
    content = f.read()
    f.close()
    allow = content[11:12]

    if allow == '1':  # 如果允许插入

        f = open('allowinsertdatabase.txt', 'w')
        f.write('is_allowed=0')
        f.close()

        print('准备插入数据库')

        event_type_int = int(event_type) if event_type else None
        old_people_id_int = int(old_people_id) if old_people_id else None
        event_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # payload = {'id': 0,  # id=0 means insert; id=1 means update;
        payload = {'event_desc': event_desc,
                   'event_type': event_type_int,
                   'event_date': event_date,
                   'event_location': event_location,
                   'oldperson_id': old_people_id_int}
        print(payload)

        headers = {'content-type': 'application/json'}
        ret = requests.post(url, json=payload, headers=headers)
        if ret.status_code == 200:
            text = json.loads(ret.text)
            print(text)
            if text['code'] == 1:
                print('插入成功')
            else:
                print('插入失败')
        else:
            print('error')

    else:
        print('等待中')
