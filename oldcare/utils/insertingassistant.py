import os
import datetime
import cv2
from oldcare.utils import communicationassistant
from oldcare.utils.pathassistant import get_path


def inserting(event_desc, event_type, event_location, old_people_id, output_path, frame):
    url = "http://localhost:10000/eventInfo/addEvent"

    insert_control_file_path = get_path('insert_control_file_path')

    f = open(insert_control_file_path, 'r')
    content = f.read()
    f.close()
    allow = content[11:12]

    if allow == '1':  # 如果允许插入
        f = open(insert_control_file_path, 'w')
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

        path = ''
        response = communicationassistant.get_response(url, payload)
        if response == 'error':
            print('error')
        else:
            if response['code'] == 1:
                path = response['msg']
                print('插入成功')
            else:
                print('插入失败')

        print(os.path.join(output_path, path, 'snapshot.jpg'))
        cv2.imwrite(os.path.join(output_path, path, 'snapshot.jpg'), frame)
    else:
        print('等待中')
