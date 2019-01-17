import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import OrderedDict
import seaborn as sns
import geopy.distance

MIN_SPEED = 4
MIN_HEARTRATE = 0
GPS_TIME_DIFF = 1 # hours
TIMESLOT_LEN = 0.015

ROOT_DN = 1
R1_DN = 5376
R2_DN = 2697
R3_DN = 5392

STATIONARY_NODES = {}
STATIONARY_NODES[ROOT_DN] = (50.853522, 4.711134)
STATIONARY_NODES[R1_DN] = (50.853785, 4.712408)
STATIONARY_NODES[R2_DN] = (50.853128, 4.713181)
STATIONARY_NODES[R3_DN] = (50.854576, 4.711852)

REBOOT_THRESHOLD = 100 # packets

def parseMultihop(RSSIList):
    count = []

def parseBytes(lineBytes):
    logTimestamp = lineBytes.split(' [CONAMO:INFO] ')[0]
    # convert timestamp to object
    logTimestampObject = datetime.strptime(logTimestamp, "%Y-%m-%d %H:%M:%S,%f")
    # convert a string represting a list to an actul list
    bytes = lineBytes.split(' [CONAMO:INFO] ')[1].split('[')[1].split(']')[0].split(',')
    # convert it to an int (whitespace will be removed automatically)
    bytes = [int(byte) for byte in bytes]

    data = {}
    data['deviceUUID'] = bytes[9] * 16**2 + bytes[10]
    data['sensorUUID'] = bytes[20] * 16**2 + bytes[21]
    data['gpsTimestamp'] = bytes[16] * 16**6 + bytes[17] * 16**4 + bytes[18] * 16**2 + bytes[19]
    data['counter'] = bytes[11] * 16**2 + bytes[12]
    data['speed'] = bytes[34] * 16**2 + bytes[35]
    data['heartrate'] = bytes[22]
    data['location'] = {}
    # print 'LAT %d-%d-%d-%d' % (bytes[24], bytes[25], bytes[26], bytes[27])
    # print 'LAT 2 %d-%d-%d-%d' % (bytes[24] * 16**6, bytes[25] * 16**4, bytes[26] * 16**2, bytes[27])
    # print 'LAT 3 %d' % (bytes[24] * 16**6 + bytes[25] * 16**4 + bytes[26] * 16**2 + bytes[27])
    # print 'LON %d-%d-%d-%d' % (bytes[28], bytes[29], bytes[30], bytes[31])
    # print 'LON 2 %d-%d-%d-%d' % (bytes[28] * 16**6, bytes[29] * 16**4, bytes[30] * 16**2, bytes[31])
    # print 'LON 3 %d' % (bytes[28] * 16**6 + bytes[29] * 16**4 + bytes[30] * 16**2 + bytes[31])
    data['location']['lon'] = bytes[28] * 16**6 + bytes[29] * 16**4 + bytes[30] * 16**2 + bytes[31]
    data['location']['lat'] = bytes[24] * 16**6 + bytes[25] * 16**4 + bytes[26] * 16**2 + bytes[27]
    data['location']['nsew'] = bytes[23]
    data['ASNarrival'] = bytes[6] * 16**8 + bytes[5] * 16**6 + bytes[4] * 16**4 + bytes[3] * 16**2 + bytes[2]
    data['ASNinit'] = bytes[13] * 16**4 + bytes[14] * 16**2 + bytes[15]

    RSSI = []
    val = bytes[40]
    if (val & (1 << (8 - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << 8)        # compute negative value
    if val != 0:
        RSSI.append(val)
    val = bytes[46]
    if (val & (1 << (8 - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << 8)        # compute negative value
    if val != 0:
        RSSI.append(val)
    val = bytes[52]
    if (val & (1 << (8 - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << 8)        # compute negative value
    if val != 0:
        RSSI.append(val)

    data['RSSI'] = RSSI
    data['multihop'] = len(data['RSSI'])
    if data['multihop'] > 3:
        assert False

    LATENCY = []
    LATENCY.append((bytes[41] * 16**4 + bytes[42] * 16**2 + bytes[43]) - data['ASNinit'])
    LATENCY.append(((bytes[47] * 16**4 + bytes[48] * 16**2 + bytes[49]) - data['ASNinit']) - LATENCY[-1])
    data['LATENCIES'] = []
    for index in range(data['multihop'] - 1):
        data['LATENCIES'].append(LATENCY[index])
    if data['multihop'] > 1:
        data['LATENCIES'].append((data['ASNarrival'] - data['ASNinit']) - data['LATENCIES'][-1])
    else:
        data['LATENCIES'].append(data['ASNarrival'] - data['ASNinit'])

    PARENTS = []
    PARENTS.append(bytes[38] * 16**2 + bytes[39])
    PARENTS.append(bytes[44] * 16**2 + bytes[45])
    PARENTS.append(bytes[50] * 16 ** 2 + bytes[51])
    data['PARENTS'] = []
    for index in range(data['multihop'] - 1):
        data['PARENTS'].append(PARENTS[index])
    data['PARENTS'].append(ROOT_DN)
    # if data['multihop'] > 1:
    #     data['LATENCIES'].append((data['ASNarrival'] - data['ASNinit']) - data['LATENCIES'][-1])
    # else:
    #     data['LATENCIES'].append(data['ASNarrival'] - data['ASNinit'])

# if there was a timestamp, convert it to a timestamp
    data['gpsTime'] = True
    if data['gpsTimestamp'] != 0:
        hours = data['gpsTimestamp'] / 10000000
        minutes = (data['gpsTimestamp'] - hours * 10000000) / 100000
        seconds = (data['gpsTimestamp'] - hours * 10000000 - minutes * 100000) / 1000
        microseconds = (data['gpsTimestamp'] - hours * 10000000 - minutes * 100000 - seconds * 1000) * 1000
        data['gpsTimestamp'] = datetime(logTimestampObject.year, \
                                        logTimestampObject.month, \
                                        logTimestampObject.day, \
                                        hours, minutes, seconds, microseconds)
        data['gpsTimestamp'] = data['gpsTimestamp'] + timedelta(hours=GPS_TIME_DIFF)
    else:
        data['gpsTime'] = False

    if data['location']['lon'] != 0:
        latDegrees = math.floor(data['location']['lat'] / 1000000);
        latMinutes = (data['location']['lat'] - (latDegrees * 1000000)) / 10000;
        data['location']['lat'] = latDegrees + (latMinutes / 60);
        lonDegrees = math.floor(data['location']['lon'] / 1000000);
        lonMinutes = (data['location']['lon'] - (lonDegrees * 1000000)) / 10000;
        data['location']['lon'] = lonDegrees + (lonMinutes / 60);
    #
    # print 'lon %.10f' % data['location']['lon']
    # print 'lat %.10f' % data['location']['lat']

    # print data

    return data['deviceUUID'], logTimestampObject, data


def parseJSON(lineJSON):
    logTimestamp = lineJSON.split(' [CONAMO:INFO] ')[0]
    # convert timestamp to object
    logTimestampObject = datetime.strptime(logTimestamp, "%Y-%m-%d %H:%M:%S,%f")
    data = json.loads(lineJSON.split(' [CONAMO:INFO] ')[1].replace("'", '"'))
    deviceUUID = data['deviceUUID']

    if data['location']['lon'] != 0:
        latDegrees = math.floor(data['location']['lat'] / 1000000);
        latMinutes = (data['location']['lat'] - (latDegrees * 1000000)) / 10000;
        data['location']['lat'] = latDegrees + (latMinutes / 60);
        lonDegrees = math.floor(data['location']['lon'] / 1000000);
        lonMinutes = (data['location']['lon'] - (lonDegrees * 1000000)) / 10000;
        data['location']['lon'] = lonMinutes + (lonMinutes / 60);

    return data['deviceUUID'], logTimestampObject, data


def parseFile(file, data, filterStart, filterStop):
    with open(file) as f:
        for line in f:
             # strip from whitespace
            line = line.strip()

            dataLine = None
            deviceUUID = None
            # parse the data
            if 'deviceUUID' in line:
                deviceUUID, logTimestamp, dataline = parseJSON(line)
            else:
                deviceUUID, logTimestamp, dataLine = parseBytes(line)

            if filterStart <= logTimestamp <= filterStop:
                # only restrict data to persons we know
                if deviceUUID in persons:
                    # dataLine[0] contains the deviceUUID
                    if deviceUUID not in data:
                        data[deviceUUID] = {'json': OrderedDict(), 'bytes': OrderedDict()}

                    # add the data
                    if 'deviceUUID' in line:
                        data[deviceUUID]['json'][logTimestamp] = dataLine
                        data[0]['json'][logTimestamp] = dataLine
                    else:
                        data[deviceUUID]['bytes'][logTimestamp] = dataLine
                        data[0]['bytes'][logTimestamp] = dataLine

def generateHTMLOverview(deviceUUID, data, persons):
    counter = 0
    hrCounter = 0
    avgHR = 0
    minHR = 99999
    maxHR = 0
    speedCounter = 0
    minSpeed = 99999
    maxSpeed = 0
    avgSpeed = 0
    multihopCounter = 0

    timeLatencies = []
    avgTimeLatency = 'N/A'

    ASNLatencies = []

    ASNLatenciesDirect = []
    ASNLatenciesMultihop = []

    html = '<html><head><title>{name}</title></head><body style="font-size:95%">'.format(name=persons[deviceUUID])
    html += '<h1>Data on {name}</h1>'.format(name=persons[deviceUUID])
    table = '<table border="1" style="font-size:90%"><tr><th>datapoint</th><th>pktCounter</th><th>deviceUUID</th><th>logged at root</th><th>generated at GPS time</th><th>time latency (s)</th><th>location</th><th>multihop</th><th>Parents</th><th>RSSI to parent</th><th>Latency to parent</th><th>ASN at source</th><th>ASN arrival at root</th><th>Latency (ASNs)</th><th>ASN latency (in seconds)</th><th>heartrate</th><th>speed</th></tr>'

    # print data
    # print deviceUUID
    for logTimestamp, datapoint in data[deviceUUID]['bytes'].iteritems():
        location = '<a href="http://maps.google.com/maps?z=25&q={lat},{lon}&t=k" target="_blank">location</a>'.format(lat=str(datapoint['location']['lat']), lon=str(datapoint['location']['lon']))
        tdSpeed = '<td style="background-color: #ffad99">'
        if datapoint['speed'] >= MIN_SPEED:
            tdSpeed = '<td style="background-color: LightGreen">'
        tdHeartrate = '<td style="background-color: #ffad99">'
        if datapoint['heartrate'] > MIN_HEARTRATE:
            tdHeartrate = '<td style="background-color: LightGreen">'
        tdMultihop = '<td>'
        if datapoint['multihop'] > 2:
            tdMultihop = '<td style="background-color: LightBlue">'

        timeLatency = 'N/A'
        if datapoint['gpsTime']:
            timeLatency = (logTimestamp - datapoint['gpsTimestamp']).seconds
            timeLatencies.append(timeLatency)

        ASNLatency = datapoint['ASNarrival'] - datapoint['ASNinit']
        ASNLatencies.append(ASNLatency)
        if ASNLatency < 0:
            raise -1

        if datapoint['multihop'] > 1:
            ASNLatenciesMultihop.append(ASNLatency)
        else:
            ASNLatenciesDirect.append(ASNLatency)

        speed = datapoint['speed']

        table += '<tr><td>{counter}</td><td>{pktCounter}</td><td>{deviceUUID}</td><td>{logTimestamp}</td><td>{gpsTimestamp}</td><td>{timeLatency}</td>\
        <td>{location}</td>{tdMultihop}{multihop}</td><td>{parents}</td><td>{rssi}</td><td>{latencies}</td><td>{asninit}</td><td>{asnarrival}</td><td>{latencyASN}</td><td>{latencysec}</td>{tdhr}{heartrate}</td>{tdspeed}{speed}</td></tr>'.format(counter=counter,\
         pktCounter=datapoint['counter'], \
         deviceUUID=datapoint['deviceUUID'], \
         logTimestamp=str(logTimestamp), \
         gpsTimestamp=str(datapoint['gpsTimestamp']), \
         timeLatency=timeLatency,  \
         location=location, \
         tdMultihop=tdMultihop, \
         multihop=datapoint['multihop'], \
         parents=str(datapoint['PARENTS']),\
         rssi=str(datapoint['RSSI']), \
         latencies=str(datapoint['LATENCIES']), \
         asninit=datapoint['ASNinit'], \
         asnarrival=datapoint['ASNarrival'], \
         latencyASN=ASNLatency, \
         latencysec=round(float(ASNLatency) * TIMESLOT_LEN,2), \
         tdhr=tdHeartrate, \
         heartrate=datapoint['heartrate'], \
         tdspeed=tdSpeed, \
         speed=speed)

        if datapoint['heartrate'] > MIN_HEARTRATE:
            hrCounter += 1
            avgHR += datapoint['heartrate']
            if datapoint['heartrate'] > maxHR:
                maxHR = datapoint['heartrate']
            if datapoint['heartrate'] < minHR:
                minHR = datapoint['heartrate']

        if datapoint['speed'] >= MIN_SPEED:
            speedCounter += 1
            avgSpeed += datapoint['speed']
            if datapoint['speed'] > maxSpeed:
                maxSpeed = datapoint['speed']
            if datapoint['speed'] < minSpeed:
                minSpeed = datapoint['speed']

        if datapoint['multihop'] == True:
            multihopCounter += 1

        counter += 1

    table += '</table></body></html>'
    html += 'Total number of datapoints: <b>{counter}</b><br /><br />'.format(counter=counter)
    if hrCounter > 0:
        avgHR = avgHR/float(hrCounter)
    else:
        avgHR = 'N/A'
    html += 'Total number of heartrate datapoints (i.e, hr > 0): <b>{hrCounter}</b><br />'.format(hrCounter=hrCounter)
    html += 'Average heartrate (with hr > 0): <b>{avgHR}</b><br />'.format(avgHR=avgHR)
    html += 'Min heartrate (with hr > 0): <b>{minHR}</b><br />'.format(minHR=minHR)
    html += 'Max heartrate (with hr > 0): <b>{maxHR}</b><br /><br />'.format(maxHR=maxHR)
    if speedCounter > 0:
        avgSpeed = avgSpeed/float(speedCounter)
    else:
        avgSpeed = 'N/A'
    html += 'Total number of (moving) speed datapoints (i.e, we define <u>moving</u> as >= 4 kph): <b>{hrCounter}</b><br />'.format(hrCounter=speedCounter)
    html += 'Average speed (with speed >= 4): <b>{avgSpeed}</b><br />'.format(avgSpeed=avgSpeed)
    html += 'Min speed (with speed >= 4): <b>{minSpeed}</b><br />'.format(minSpeed=minSpeed)
    html += 'Max speed (with speed >= 4): <b>{maxSpeed}</b><br /><br />'.format(maxSpeed=maxSpeed)

    html += 'Total number of multihop datapoints: <b>{multihopCounter}</b> ({percentage}%)<br /><br />'.format(multihopCounter=multihopCounter, percentage=round((multihopCounter/float(counter))*100, 2))
    if len(timeLatencies) > 0:
        html += 'Average time latency: <b>{avg}</b> (std = {std}) ({nr} datapoints)<br />'.format(avg=np.mean(timeLatencies), std=np.std(timeLatencies), nr=len(timeLatencies))
        html += 'Min time latency: <b>{minlst}</b> s<br />'.format(minlst=min(timeLatencies))
        html += 'Max time latency: <b>{maxlst}</b> s<br /><br />'.format(maxlst=max(timeLatencies))

    html += 'Average ASN latency: <b>{avg}</b> (std = {std}) ({nr} datapoints)<br />'.format(avg=np.mean(ASNLatencies)*TIMESLOT_LEN, std=np.std(ASNLatencies)*TIMESLOT_LEN, nr=len(ASNLatencies))
    html += 'Min ASN latency: <b>{minlst}</b> s<br />'.format(minlst=min(ASNLatencies)*TIMESLOT_LEN)
    html += 'Max ASN latency: <b>{maxlst}</b> s<br /><br />'.format(maxlst=max(ASNLatencies)*TIMESLOT_LEN)

    if len(ASNLatenciesDirect) > 0:
        html += 'Average <u>direct</u> link ASN latency: <b>{avg}</b> (std = {std}) ({nr} datapoints)<br />'.format(avg=np.mean(ASNLatenciesDirect)*TIMESLOT_LEN, std=np.std(ASNLatenciesDirect)*TIMESLOT_LEN, nr=len(ASNLatenciesDirect))
        html += 'Min <u>direct</u> link ASN latency: <b>{minlst}</b> s<br />'.format(minlst=min(ASNLatenciesDirect)*TIMESLOT_LEN)
        html += 'Max <u>direct</u> link ASN latency: <b>{maxlst}</b> s<br /><br />'.format(maxlst=max(ASNLatenciesDirect)*TIMESLOT_LEN)

    if len(ASNLatenciesMultihop) > 0:
        html += 'Average <u>multihop</u> ASN latency: <b>{avg}</b> (std = {std}) ({nr} datapoints)<br />'.format(avg=np.mean(ASNLatenciesMultihop)*TIMESLOT_LEN, std=np.std(ASNLatenciesMultihop)*TIMESLOT_LEN, nr=len(ASNLatenciesMultihop))
        html += 'Min <u>multihop</u> ASN latency: <b>{minlst}</b> s<br />'.format(minlst=min(ASNLatenciesMultihop)*TIMESLOT_LEN)
        html += 'Max <u>multihop</u> ASN latency: <b>{maxlst}</b> s<br /><br />'.format(maxlst=max(ASNLatenciesMultihop)*TIMESLOT_LEN)

    html += table

    fName = 'html/{deviceUUID}-{name}-overview.html'.format(deviceUUID=deviceUUID, name=persons[deviceUUID])
    fo = open(fName, 'w')
    fo.write(html)
    fo.close()
    # print html

def plotLatencyScatter(data):
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    colors = ['r', 'g', 'c', 'm', 'y']

    # x and y data per # of multihops
    plotData = {\
                1: {'x': [], 'y': [], 'color': [], 'area': []}, \
                2: {'x': [], 'y': [], 'color': [], 'area': []}, \
                3: {'x': [], 'y': [], 'color': [], 'area': []}, \
                4: {'x': [], 'y': [], 'color': [], 'area': []}, \
                5: {'x': [], 'y': [], 'color': [], 'area': []} \
                }

    validCount = 0
    outlierCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        # we filter the data for 70000, apperantly some outlier data
        if datapoint['gpsTime'] and (logTimestamp - datapoint['gpsTimestamp']).seconds < 50000:
            plotData[datapoint['multihop']]['x'].append((logTimestamp - datapoint['gpsTimestamp']).seconds) # time latency
            plotData[datapoint['multihop']]['y'].append(round(float(datapoint['ASNarrival'] - datapoint['ASNinit']) * TIMESLOT_LEN,2)) # ASN latency
            plotData[datapoint['multihop']]['color'].append(colors[(datapoint['multihop']-1)])
            plotData[datapoint['multihop']]['area'].append(7)
            validCount += 1
        elif datapoint['gpsTime'] and (logTimestamp - datapoint['gpsTimestamp']).seconds > 50000:
            outlierCount += 1

    # N = 50
    # x = np.random.rand(N)
    # y = np.random.rand(N)
    # colors = np.random.rand(N)
    # area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii
    fig = plt.figure()
    plt.subplot(231)
    plt.scatter(plotData[1]['x'], plotData[1]['y'], s=plotData[1]['area'], c=plotData[1]['color'], alpha=0.5)
    plt.subplot(232)
    plt.scatter(plotData[2]['x'], plotData[2]['y'], s=plotData[2]['area'], c=plotData[2]['color'], alpha=0.5)
    plt.subplot(233)
    plt.scatter(plotData[3]['x'], plotData[3]['y'], s=plotData[3]['area'], c=plotData[3]['color'], alpha=0.5)
    plt.subplot(234)
    plt.scatter(plotData[4]['x'], plotData[4]['y'], s=plotData[4]['area'], c=plotData[4]['color'], alpha=0.5)
    plt.subplot(235)
    plt.scatter(plotData[5]['x'], plotData[5]['y'], s=plotData[5]['area'], c=plotData[5]['color'], alpha=0.5)
    plt.subplot(236)
    plt.scatter(plotData[1]['x'], plotData[1]['y'], s=plotData[1]['area'], c=plotData[1]['color'], alpha=0.5)
    plt.scatter(plotData[2]['x'], plotData[2]['y'], s=plotData[2]['area'], c=plotData[2]['color'], alpha=0.5)
    plt.scatter(plotData[3]['x'], plotData[3]['y'], s=plotData[3]['area'], c=plotData[3]['color'], alpha=0.5)
    plt.scatter(plotData[4]['x'], plotData[4]['y'], s=plotData[4]['area'], c=plotData[4]['color'], alpha=0.5)
    plt.scatter(plotData[5]['x'], plotData[5]['y'], s=plotData[5]['area'], c=plotData[5]['color'], alpha=0.5)
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.ylabel('TSCH slot latency (seconds)')
    plt.xlabel('GPS time latency (seconds)')
    fig.savefig("html/plotLatencyScatter.pdf", bbox_inches='tight')
    plt.close()

    print 'There are {valid} datapoints and {outliers} outliers.'.format(valid=validCount, outliers=outlierCount)

def plotLatencyScatterPerHop(data, method='mean'):
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    colors = ['r', 'g', 'c', 'm', 'y']

    # x and y data per # of multihops
    plotData = {\
                1: {'x': [], 'y': [], 'color': [], 'area': []}, \
                2: {'x': [], 'y': [], 'color': [], 'area': []}, \
                3: {'x': [], 'y': [], 'color': [], 'area': []}, \
                }

    validCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        tmpRSSI = [rssi for rssi in datapoint['RSSI'] if rssi != 0]
        if method == 'mean':
            tmpRSSI = np.mean(tmpRSSI)
        elif method == 'max':
            tmpRSSI = max(tmpRSSI)
        elif method == 'min':
            tmpRSSI = min(tmpRSSI)
        plotData[datapoint['multihop']]['x'].append(tmpRSSI)  # time latency
        plotData[datapoint['multihop']]['y'].append(round(float(datapoint['ASNarrival'] - datapoint['ASNinit']) * TIMESLOT_LEN,2)) # ASN latency
        plotData[datapoint['multihop']]['color'].append(colors[(datapoint['multihop']-1)])
        plotData[datapoint['multihop']]['area'].append(7)
        validCount += 1

    # N = 50
    # x = np.random.rand(N)
    # y = np.random.rand(N)
    # colors = np.random.rand(N)
    # area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii
    fig = plt.figure(figsize=(14,9))
    ax1 = plt.subplot(131)
    # plt.ylabel('TSCH slot latency (seconds)')
    # plt.xlabel('Average RSSI over all hops (dBm)')
    plt.scatter(plotData[1]['x'], plotData[1]['y'], s=plotData[1]['area'], c=plotData[1]['color'], alpha=0.5)
    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
    # plt.ylabel('TSCH slot latency (seconds)')
    # plt.xlabel('Average RSSI over all hops (dBm)')
    plt.scatter(plotData[2]['x'], plotData[2]['y'], s=plotData[2]['area'], c=plotData[2]['color'], alpha=0.5)
    ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
    # plt.ylabel('TSCH slot latency (seconds)')
    # plt.xlabel('Average RSSI over all hops (dBm)')
    plt.scatter(plotData[3]['x'], plotData[3]['y'], s=plotData[3]['area'], c=plotData[3]['color'], alpha=0.5)
    # ax4 = plt.subplot(134, sharex=ax1, sharey=ax1)
    # plt.ylabel('TSCH slot latency (seconds)')
    # plt.xlabel('Average RSSI over all hops (dBm)')
    # plt.scatter(plotData[4]['x'], plotData[4]['y'], s=plotData[4]['area'], c=plotData[4]['color'], alpha=0.5)
    # ax5 = plt.subplot(235, sharex=ax1, sharey=ax1)
    # # plt.ylabel('TSCH slot latency (seconds)')
    # # plt.xlabel('Average RSSI over all hops (dBm)')
    # plt.scatter(plotData[5]['x'], plotData[5]['y'], s=plotData[5]['area'], c=plotData[5]['color'], alpha=0.5)
    # ax6 = plt.subplot(236, sharex=ax1, sharey=ax1)
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    # plt.ylabel('TSCH slot latency (seconds)')
    # plt.xlabel('Average RSSI over all hops (dBm)')
    fig.text(0.5, 0.0005, 'Average RSSI over all hops (dBm)', ha='center')
    fig.text(0.0005, 0.5, 'TSCH slot latency (seconds)', va='center', rotation='vertical')
    fig.tight_layout()
    name = 'html/plotLatencyScatterPerHop-%s.pdf' % method
    fig.savefig(name)
    plt.close()

    print 'There are {valid} datapoints.'.format(valid=validCount)

def plotRSSILatencyScatter(data):
    colors = ['r', 'g', 'c', 'm', 'y']

    # x and y data per # of multihops
    plotData = {'x': [], 'y': [], 'color': [], 'area': []}

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        for ix, rssi in enumerate(datapoint['RSSI']):
            plotData['x'].append(rssi) # time latency
            plotData['y'].append(round(float(datapoint['LATENCIES'][ix]) * TIMESLOT_LEN,2)) # ASN latency
            plotData['color'].append(5)
            plotData['area'].append(2)

    fig = plt.figure()
    plt.scatter(plotData['x'], plotData['y'], s=plotData['area'], c=plotData['color'], alpha=0.3)
    plt.ylabel('TSCH slot latency (seconds)')
    plt.xlabel('RSSI (dBm)')
    fig.savefig("html/plotRSSILatencyScatter.pdf", bbox_inches='tight')
    plt.close()

def plotRSSILatencyHeatmap(data):
    # x and y data per # of multihops
    plotData = {'x': [], 'y': [], 'color': [], 'area': []}

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        for ix, rssi in enumerate(datapoint['RSSI']):
            lat = round(float(datapoint['LATENCIES'][ix]) * TIMESLOT_LEN, 2)
            if lat <= 10:
                plotData['x'].append(rssi) # time latency
                plotData['y'].append(lat) # ASN latency
                plotData['color'].append(5)
                plotData['area'].append(2)

    # fig = plt.figure()
    # heatmap, xedges, yedges = np.histogram2d(plotData['x'], plotData['y'], bins=(200, 100))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    h = sns.jointplot(plotData['x'], plotData['y'], kind='hex')
    h.set_axis_labels('RSSI (dBm)', 'TSCH slot latency (seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig("html/plotRSSILatencyHeatmap.pdf")
    plt.close()

def plotDistanceLatencyHeatmap(data):
    # x and y data per # of multihops
    plotData = {'x': [], 'y': [], 'color': [], 'area': []}

    validCount = 0
    notValidCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        lat = datapoint['location']['lat']
        lon = datapoint['location']['lon']
        parent = datapoint['PARENTS'][0]
        latency = round(float(datapoint['LATENCIES'][0]) * TIMESLOT_LEN, 2)
        if latency <= 2:
            if parent not in STATIONARY_NODES:
                print parent
                print STATIONARY_NODES
                assert False
            if lat != 0 and lon != 0:
                plotData['x'].append(calcMeters(lat, lon, STATIONARY_NODES[parent][0], STATIONARY_NODES[parent][1])) # ASN latency
                plotData['y'].append(latency)
                plotData['color'].append(5)
                plotData['area'].append(2)
                validCount += 1
            else:
                notValidCount += 1

    # fig = plt.figure()
    # heatmap, xedges, yedges = np.histogram2d(plotData['x'], plotData['y'], bins=(200, 100))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    h = sns.jointplot(plotData['x'], plotData['y'], kind='hex')
    h.set_axis_labels('Distance (meters)', 'TSCH slot latency (seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig("html/plotDistanceLatencyHeatmap.pdf")
    plt.close()

def plotDistanceRSSIHeatmap(data):
    # x and y data per # of multihops
    plotData = {'x': [], 'y': [], 'color': [], 'area': []}

    validCount = 0
    notValidCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        lat = datapoint['location']['lat']
        lon = datapoint['location']['lon']
        parent = datapoint['PARENTS'][0]
        rssi = datapoint['RSSI'][0]
        if parent not in STATIONARY_NODES:
            print parent
            print STATIONARY_NODES
            assert False
        if lat != 0 and lon != 0:
            plotData['x'].append(calcMeters(lat, lon, STATIONARY_NODES[parent][0], STATIONARY_NODES[parent][1])) # ASN latency
            plotData['y'].append(rssi)
            plotData['color'].append(5)
            plotData['area'].append(2)
            validCount += 1
        else:
            notValidCount += 1

    # fig = plt.figure()
    # heatmap, xedges, yedges = np.histogram2d(plotData['x'], plotData['y'], bins=(200, 100))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    h = sns.jointplot(plotData['x'], plotData['y'], kind='hex')
    h.set_axis_labels('Distance (meters)', 'RSSI (dBm)', fontsize=12)
    plt.tight_layout()
    plt.savefig("html/plotDistanceRSSIHeatmap.pdf")
    plt.close()

def plotDistanceLatencyScatter(data):
    # x and y data per # of multihops
    plotData = {'x': [], 'y': [], 'color': [], 'area': []}

    validCount = 0
    notValidCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        lat = datapoint['location']['lat']
        lon = datapoint['location']['lon']
        parent = datapoint['PARENTS'][0]
        latency = round(float(datapoint['LATENCIES'][0]) * TIMESLOT_LEN, 2)
        if parent not in STATIONARY_NODES:
            print parent
            print STATIONARY_NODES
            assert False
        if lat != 0 and lon != 0:
            plotData['x'].append(calcMeters(lat, lon, STATIONARY_NODES[parent][0], STATIONARY_NODES[parent][1])) # ASN latency
            plotData['y'].append(latency)
            plotData['color'].append(5)
            plotData['area'].append(2)
            validCount += 1
        else:
            notValidCount += 1

    # fig = plt.figure()
    # heatmap, xedges, yedges = np.histogram2d(plotData['x'], plotData['y'], bins=(200, 100))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.scatter(plotData['x'], plotData['y'], s=plotData['area'], c=plotData['color'], alpha=0.3)
    plt.ylabel('TSCH slot latency (seconds)')
    plt.xlabel('Distance (meters)')
    plt.savefig("html/plotDistanceLatencyScatter.pdf")
    plt.close()

def plotDistanceRSSIScatter(data):
    # x and y data per # of multihops
    plotData = {'x': [], 'y': [], 'color': [], 'area': []}

    validCount = 0
    notValidCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        lat = datapoint['location']['lat']
        lon = datapoint['location']['lon']
        parent = datapoint['PARENTS'][0]
        rssi = datapoint['RSSI'][0]
        if parent not in STATIONARY_NODES:
            print parent
            print STATIONARY_NODES
            assert False
        if lat != 0 and lon != 0:
            plotData['x'].append(calcMeters(lat, lon, STATIONARY_NODES[parent][0], STATIONARY_NODES[parent][1])) # ASN latency
            plotData['y'].append(rssi)
            plotData['color'].append(5)
            plotData['area'].append(2)
            validCount += 1
        else:
            notValidCount += 1

    # fig = plt.figure()
    # heatmap, xedges, yedges = np.histogram2d(plotData['x'], plotData['y'], bins=(200, 100))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.scatter(plotData['x'], plotData['y'], s=plotData['area'], c=plotData['color'], alpha=0.3)
    plt.ylabel('RSSI (dBm)')
    plt.xlabel('Distance (meters)')
    plt.savefig("html/plotDistanceRSSIScatter.pdf")
    plt.close()

def plotLatencyBars(data):
    # x and y data per # of multihops
    plotData = {\
                1: {'gpslatency': [], 'latency': [], 'color': [], 'area': [], 'count': 0}, \
                2: {'gpslatency': [], 'latency': [], 'color': [], 'area': [], 'count': 0}, \
                3: {'gpslatency': [], 'latency': [], 'color': [], 'area': [], 'count': 0}, \
                }

    validCount = 0
    outlierCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        # we filter the data for 70000, apperantly some outlier data
        if datapoint['gpsTime'] and (logTimestamp - datapoint['gpsTimestamp']).seconds < 50000:
            plotData[datapoint['multihop']]['gpslatency'].append((logTimestamp - datapoint['gpsTimestamp']).seconds) # time latency
            plotData[datapoint['multihop']]['latency'].append(round(float(datapoint['ASNarrival'] - datapoint['ASNinit']) * TIMESLOT_LEN,2)) # ASN latency
            plotData[datapoint['multihop']]['color'].append(0)
            plotData[datapoint['multihop']]['area'].append(7)
            plotData[datapoint['multihop']]['count'] += 1
            validCount += 1
        elif datapoint['gpsTime'] and (logTimestamp - datapoint['gpsTimestamp']).seconds > 50000:
            outlierCount += 1

    fig, ax = plt.subplots()

    dat1 = [plotData[1]['latency'], plotData[2]['latency'], plotData[3]['latency']]
    dat1 = [np.mean(l) for l in dat1]
    dat2 = [plotData[1]['gpslatency'], plotData[2]['gpslatency'], plotData[3]['gpslatency']]
    dat2 = [np.mean(l) for l in dat2]

    one = ax.bar([1,2,3], dat1, color='g', width=0.4)
    one_one = ax.bar([1+0.4,2+0.4,3+0.4], dat2, color='c', width=0.4)

    ax.legend((one, one_one), ('Slot Latency', 'GPS Latency'))
    ax.set_ylabel('Latency (seconds)')
    ax.set_xlabel('Number of hops')
    fig.savefig("html/plotLatencyBars.pdf", bbox_inches='tight')
    plt.close()

    # for hopcount in plotData:
    #     print 'For {hopcount}: {count} datapoints'.format(hopcount=hopcount, count=plotData[hopcount]['count'])

    print 'There are {valid} datapoints and {outliers} outliers.'.format(valid=validCount, outliers=outlierCount)

def plotLatencyBoxplots(data):
    # x and y data per # of multihops
    plotData = {\
                1: {'x': [], 'y': [], 'color': [], 'area': [], 'count': 0}, \
                2: {'x': [], 'y': [], 'color': [], 'area': [], 'count': 0}, \
                3: {'x': [], 'y': [], 'color': [], 'area': [], 'count': 0}, \
                }

    validCount = 0
    outlierCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        # we filter the data for 70000, apperantly some outlier data
        # if datapoint['gpsTime'] and (logTimestamp - datapoint['gpsTimestamp']).seconds < 50000:
        # plotData[datapoint['multihop']]['x'].append((logTimestamp - datapoint['gpsTimestamp']).seconds) # time latency
        plotData[datapoint['multihop']]['y'].append(round(float(datapoint['ASNarrival'] - datapoint['ASNinit']) * TIMESLOT_LEN,2)) # ASN latency
        plotData[datapoint['multihop']]['color'].append(0)
        plotData[datapoint['multihop']]['area'].append(7)
        plotData[datapoint['multihop']]['count'] += 1
        #     validCount += 1
        # elif datapoint['gpsTime'] and (logTimestamp - datapoint['gpsTimestamp']).seconds > 50000:
        #     outlierCount += 1


    fig, ax = plt.subplots()
    dat = [plotData[1]['y'], plotData[2]['y'], plotData[3]['y']]
    means = []
    for hp in plotData:
        iqr = np.percentile(plotData[hp]['y'], 75) - np.percentile(plotData[hp]['y'], 25)
        datMean = []
        for elem in plotData[hp]['y']:
            if elem >= (np.percentile(plotData[hp]['y'], 25) - iqr * 1.5) and elem <= (np.percentile(plotData[hp]['y'], 75) + iqr * 1.5):
                datMean.append(elem)
        means.append(np.mean(datMean))

    ax.boxplot(dat, showfliers=False, showmeans=True)
    ax.plot([1,2,3], means, 'go')
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    ax.set_ylabel('TSCH Slot Latency (seconds)')
    ax.set_xlabel('Number of hops')
    # ax.set_ylim(15)
    fig.savefig("html/plotLatencyBoxplots.pdf", bbox_inches='tight')
    plt.close()

    # median = np.median(plotData[1]['y'])
    # upper_quartile = np.percentile(plotData[1]['y'], 75)
    # lower_quartile = np.percentile(plotData[1]['y'], 25)
    # iqr = upper_quartile - lower_quartile
    #
    # print median
    # print upper_quartile
    # print lower_quartile
    # print iqr

    # print 'There are {valid} datapoint and {outliers} outliers.'.format(valid=validCount, outliers=outlierCount)

    # for hopcount in plotData:
    #     print 'For {hopcount}: {count} datapoints'.format(hopcount=hopcount, count=plotData[hopcount]['count'])

def calcMeters(lat1, lon1, lat2, lon2):
    # # approximate radius of earth in km
    # from math import sin, cos, sqrt, atan2, radians
    # R = 6373.0
    #
    # lat1 = radians(lat1)
    # lon1 = radians(lon1)
    # lat2 = radians(lat2)
    # lon2 = radians(lon2)
    #
    # dlon = lon2 - lon1
    # dlat = lat2 - lat1
    #
    # a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    # c = 2 * atan2(sqrt(a), sqrt(1 - a))
    #
    # distance = R * c
    # return distance / 1000.0

    return geopy.distance.VincentyDistance((lat1, lon1), (lat2, lon2)).km * 1000.0

def plotDistanceBoxplots(data):
    # x and y data per # of multihops
    plotData = {\
                ROOT_DN: {'x': [], 'y': [], 'color': [], 'area': [], 'count': 0}, \
                R1_DN: {'x': [], 'y': [], 'color': [], 'area': [], 'count': 0}, \
                R2_DN: {'x': [], 'y': [], 'color': [], 'area': [], 'count': 0}, \
                R3_DN: {'x': [], 'y': [], 'color': [], 'area': [], 'count': 0}, \
        }

    validCount = 0
    notValidCount = 0

    for logTimestamp, datapoint in data[0]['bytes'].iteritems():
        lat = datapoint['location']['lat']
        lon = datapoint['location']['lon']
        parent = datapoint['PARENTS'][0]
        if parent not in STATIONARY_NODES:
            print parent
            print STATIONARY_NODES
            assert False
        if lat != 0 and lon != 0:
            plotData[parent]['y'].append(calcMeters(lat, lon, STATIONARY_NODES[parent][0], STATIONARY_NODES[parent][1])) # ASN latency
            plotData[parent]['color'].append(0)
            plotData[parent]['area'].append(7)
            plotData[parent]['count'] += 1
            validCount += 1
        else:
            notValidCount += 1

    fig, ax = plt.subplots()
    dat = [plotData[ROOT_DN]['y'], plotData[R1_DN]['y'], plotData[R2_DN]['y'], plotData[R3_DN]['y']]
    means = []
    for pt in plotData:
        iqr = np.percentile(plotData[pt]['y'], 75) - np.percentile(plotData[pt]['y'], 25)
        datMean = []
        for elem in plotData[pt]['y']:
            if elem >= (np.percentile(plotData[pt]['y'], 25) - iqr * 1.5) and elem <= (np.percentile(plotData[pt]['y'], 75) + iqr * 1.5):
                datMean.append(elem)
        means.append(np.mean(datMean))

    ax.boxplot(dat, showfliers=False, showmeans=True)
    ax.plot([1, 2, 3, 4], means, 'go')
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    ax.set_ylabel('Distance (meters)')
    ax.set_xlabel('Bike connected to (stationary node)')
    # ax.set_ylim(15)
    ax.set_xticklabels(['root', 'R1', 'R2', 'R3'])
    fig.savefig("html/plotDistanceBoxplots.pdf", bbox_inches='tight')
    plt.close()

    print 'There are {valid} datapoints and {notValidCount} outliers.'.format(valid=validCount, notValidCount=notValidCount)

    # for hopcount in plotData:
    #     print 'For {hopcount}: {count} datapoints'.format(hopcount=hopcount, count=plotData[hopcount]['count'])

def plotRebootBars(data):
    rebootCounter = {}
    lstDevices = []
    for device, v in data.iteritems():
        if device != 0:
            previousPktCounter = None
            rebootCounter[device] = 0
            for logTimestamp, datapoint in data[device]['bytes'].iteritems():
                if previousPktCounter != None and previousPktCounter - datapoint['counter'] > REBOOT_THRESHOLD: # we take 100 counter values as the threshold
                    rebootCounter[device] += 1
                previousPktCounter = datapoint['counter']
            lstDevices.append(device)

    fig, ax = plt.subplots()

    ax.bar(range(len(rebootCounter.values())), rebootCounter.values(), color='g')

    ax.set_ylabel('Reboots')
    ax.set_xlabel('Device')
    ax.set_xticks(range(len(rebootCounter.values())))
    ax.set_xticklabels(lstDevices, size=4)
    fig.savefig("html/plotRebootsPerDeviceBars.pdf", bbox_inches='tight')
    plt.close()

# def plotDropsBars(data):
#     rebootCounter = {}
#     lstDevices = []
#     for device, v in data.iteritems():
#         if device != 0:
#             previousPktCounter = None
#             rebootCounter[device] = 0
#             for logTimestamp, datapoint in data[device]['bytes'].iteritems():
#                 if previousPktCounter != None and previousPktCounter - datapoint['counter'] > REBOOT_THRESHOLD: # we take 100 counter values as the threshold
#                     rebootCounter[device] += 1
#                 previousPktCounter = datapoint['counter']
#             lstDevices.append(device)
#
#     fig, ax = plt.subplots()
#
#     ax.bar(range(len(rebootCounter.values())), rebootCounter.values(), color='g')
#
#     ax.set_ylabel('Packet Drops')
#     ax.set_xlabel('Device')
#     ax.set_xticks(range(len(rebootCounter.values())))
#     ax.set_xticklabels(lstDevices, size=4)
#     fig.savefig("html/plotDropsBars.pdf", bbox_inches='tight')
#     plt.close()

def main(files, persons, data, filterStart, filterStop):
    # initialize dictionary for all data
    data[0] = {'json': OrderedDict(), 'bytes': OrderedDict()}

    for file in files:
        print 'Parsing {0}...'.format(file)
        parseFile(file, data, filterStart, filterStop)

    for deviceUUID, name in persons.iteritems():
        if deviceUUID in data:
            # sort by date
            data[deviceUUID]['json'] = OrderedDict(sorted(data[deviceUUID]['json'].items()))
            data[deviceUUID]['bytes'] = OrderedDict(sorted(data[deviceUUID]['bytes'].items()))
            generateHTMLOverview(deviceUUID, data, persons)

    print 'Created all individual pages...'

    html = '<html><head><title>Data overview</title></head><body>'
    html += '<h1>Data between {0} and {1}</h1>'.format(filterStart, filterStop)
    html += '<a href="{deviceUUID}-{name}-overview.html">All data</a><br /><br />'.format(deviceUUID=0, name=persons[0])
    html += 'Individual data:'
    html += '<table>'
    for deviceUUID, person in persons.iteritems():
        if deviceUUID != 0:
            html += '<tr><td><a href="{deviceUUID}-{name}-overview.html">{name}</a></td></tr>'.format(deviceUUID=deviceUUID, name=persons[deviceUUID])
    html += '</table>'
    html += '</body></html>'
    fName = 'html/overview.html'
    fo = open(fName, 'w')
    fo.write(html)
    fo.close()

    print 'Created overview page...'

    print 'Creating LatencyBars...'
    plotLatencyBars(data)
    print 'Creating LatencyBoxplots...'
    plotLatencyBoxplots(data)
    print 'Creating RSSILatencyScatter...'
    plotRSSILatencyScatter(data)
    print 'Creating RSSILatencyHeatmap...'
    plotRSSILatencyHeatmap(data)
    print 'Creating DistanceBoxplots...'
    plotDistanceBoxplots(data)
    print 'Creating DistanceLatencyHeatmap...'
    plotDistanceLatencyHeatmap(data)
    print 'Creating DistanceLatencyScatter...'
    plotDistanceLatencyScatter(data)
    print 'Creating DistanceRSSIHeatmap...'
    plotDistanceRSSIHeatmap(data)
    print 'Creating DistanceRSSIScatter...'
    plotDistanceRSSIScatter(data)
    print 'Creating RebootBars...'
    plotRebootBars(data)

if __name__ == "__main__":
    # data logs
    files = ['closing-event-data/conamo.log', \
    'closing-event-data/conamo.1.log', \
    'closing-event-data/conamo.2.log', \
    'closing-event-data/conamo.3.log', \
    'closing-event-data/conamo.4.log', \
    'closing-event-data/conamo.5.log', \
    'closing-event-data/conamo.6.log', \
    'closing-event-data/conamo.7.log', \
    'closing-event-data/conamo.8.log', \
    'closing-event-data/conamo.9.log']

    # deviceUUIDs mapped to persons
    persons = {\
    3589: 'W1', \
    3978: 'W2', \
    3980: 'W3', \
    3725: 'W4', \
    3982: 'W5', \
    3605: 'W6', \
    3717: 'W7', \
    3616: 'W8', \
    3747: 'W9', \
    3755: 'W10', \
    3631: 'W11', \
    3633: 'W12', \
    4022: 'W13', \
    3639: 'W14', \
    3744: 'W15', \
    3655: 'W16', \
    4041: 'W17', \
    3787: 'W18', \
    3802: 'W19', \
    3695: 'W20', \
    3572: 'W21', \
    3705: 'W22', \
    3582: 'W23', \
    5376: 'R1', \
    2697: 'R2', \
    5392: 'R3', \
    0: 'AllData'}

    # deviceUUIDs mapped to their data
    data = {}

    filterStart = datetime(2018, 11, 07, 14, 45, 0)
    filterStop = datetime(2018, 11, 07, 15, 30, 0)

    main(files, persons, data, filterStart, filterStop)
