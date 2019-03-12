import os
import glob
import xml.etree.ElementTree as ET
import datetime

def xml_to_list(image_dir):
    # image_path = "/home/c-use/Desktop/main_directory/d_1/w_1/1"

    xml_list = []
    xml1_list = []
    for xml_file in glob.glob(image_dir + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # print(root)

        # xml_list.append(root.find('heading').text)

        for member in root.findall('trk'):
            # print(member)

            for each in member.findall('trkseg'):

                # print(each.text)
                for each1 in each.findall('trkpt'):
                    value =  root.find('heading').text
                    value1 = each1.attrib
                    value2 = each1.find('time').text
                             
                    xml_list.append([value,value1,value2])
#                     xml1_list.append([value,value1,value2])
    
    return xml_list
    


def reducing_xml(xml_list,image_dir):

    def utc_to_loc(timestamp):

        d=datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ") #Get your naive datetime object
        d=d.replace(tzinfo=datetime.timezone.utc) #Convert it to an aware datetime object in UTC time.
        d=d.astimezone() #Convert it to your local timezone (still aware)
        d=d.strftime("%Y-%m-%dT%H:%M:%S.%fZ") #Print it with a directive of choice
    #     return d
        return datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ")

##list to store the utc to local time in datetime datetime format
    xml_utc_to_loc = []
    for each in xml_list:
    #         print(type(each[2]))
        each[2] = utc_to_loc(each[2])
        xml_utc_to_loc.append(each[2])

# print(xml_utc_to_loc[:10])


##calculates the nearest time when given a list of times and a time to query
    def nearest(xml_list,pivot):
    #     xml_list = xml_list[:][-1]
    #     print(xml_list[:10])

        near = min(xml_list, key=lambda x: abs(x - pivot))
        return near


    def converttime(img_name,xml_list):   ###input the image name output the closest time.
    #     image_name = datetime.datetime.strptime(img_name, "%Y-%m-%dT%H:%M:%S.%fZ") ##convert it into datetime

        pivot = img_name

        x = nearest(xml_utc_to_loc,pivot)
        return x

    file_names = sorted(glob.glob(image_dir + '/*.jpg'))

    new_list1 = []

    for each in file_names:

        images = each
        images = images.split("/")
        images = images[-1]
        images = images.strip(".jpg")

        images = images.split("_")
        images = images[0]+"-"+images[1]+"-"+images[2]+"T"+images[3]+":"+images[4]+":"+images[5]+"."+images[6]+"Z"

        new_list1.append(datetime.datetime.strptime(images, "%Y-%m-%dT%H:%M:%S.%fZ"))
    ##converting filenames to the datetime format
    
    
    the_nearest_time_to_gpx = []
    import time
    for each in new_list1:
        x = converttime(each,xml_list)
        the_nearest_time_to_gpx.append(x)

    idx_of_xml = []
    for each in the_nearest_time_to_gpx:
        for each1,i in zip(xml_utc_to_loc,range(len(xml_utc_to_loc))):
            if each == each1:
                idx_of_xml.append(i)

#     print(idx_of_xml)

    return idx_of_xml


# image_dir = "/home/c-use/Desktop/main_directory/d_2/w_1/"

# list_gpx = xml_to_list(image_dir)