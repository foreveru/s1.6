import os
import time, shutil

def calc_tar_pkg_name(csv_path ,team=319, track='t6'):
    import csv
    from datetime import datetime
    with open(csv_path, 'rb') as f:
        lines = csv.reader(f)
        lines = [line for line in lines]
    frames = len(lines)
    timestart = datetime.strptime(lines[0][0][-27:-4] + '000', "%Y_%m_%d_%H_%M_%S_%f")
    timeend = datetime.strptime(lines[-1][0][-27:-4] + '000', "%Y_%m_%d_%H_%M_%S_%f")
    escape = timeend - timestart
    mseconds = int(escape.seconds * 1000 + escape.microseconds / 1000)
    day = timestart.strftime("%Y%m%d")
    return "{}_{}_{}_{}_{}.tar.gz".format(day, team, track, frames, mseconds)



def process_log_folder(log_folder):
    '''
    :param log_folder: simulator path
    :return: 
    '''
    src = os.path.join(log_folder, "Log")
    dst = os.path.join(log_folder, str(int(time.time())), "Log")
    shutil.move(src, dst)
    cur_path = os.getcwd()
    os.chdir(os.path.dirname(dst))

    import tarfile
    tar = tarfile.open(os.path.join(log_folder, calc_tar_pkg_name(os.path.join(dst, "driving_log.csv"))), "w:gz")
    tar.add("Log")
    tar.close()
    os.chdir(cur_path)


if __name__ == '__main__':
    log_folder = r'D:\My Project\Formular Trend\Simulator\formula-trend-1.0.1\formula-trend-1.0.1\Windows'
    process_log_folder(log_folder)
