from os import system, mkdir
import argparse
from os.path import isdir, isfile, join
from colorama import Fore, Style

def print_error(msg):
    print(Fore.RED + msg + Style.RESET_ALL)

def print_warn(msg):
    print(Fore.YELLOW + msg + Style.RESET_ALL)

def print_highlight(msg):
    print(Fore.GREEN + msg + Style.RESET_ALL)

def get_args():
    parser = argparse.ArgumentParser(description='TartanAir')

    parser.add_argument('--output-dir', default='./',
                        help='root directory for downloaded files')

    parser.add_argument('--rgb', action='store_true', default=False,
                        help='download rgb image')

    parser.add_argument('--depth', action='store_true', default=False,
                        help='download depth image')

    parser.add_argument('--flow', action='store_true', default=False,
                        help='download optical flow')

    parser.add_argument('--seg', action='store_true', default=False,
                        help='download segmentation image')

    parser.add_argument('--only-easy', action='store_true', default=False,
                        help='download only easy trajectories')

    parser.add_argument('--only-hard', action='store_true', default=False,
                        help='download only hard trajectories')

    parser.add_argument('--only-left', action='store_true', default=False,
                        help='download only left camera')

    parser.add_argument('--only-right', action='store_true', default=False,
                        help='download only right camera')

    parser.add_argument('--only-flow', action='store_true', default=False,
                        help='download only optical flow wo/ mask')

    parser.add_argument('--only-mask', action='store_true', default=False,
                        help='download only mask wo/ flow')

    parser.add_argument('--cloudflare', action='store_true', default=False,
                        help='download the data from Scale Foundation cloudflare')

    parser.add_argument('--unzip', action='store_true', default=False,
                        help='unzip the files after downloading')

    args = parser.parse_args()

    return args

def _help():
    print ('')

class AirLabDownloader(object):
    def __init__(self, bucket_name = 'tartanair') -> None:
        from minio import Minio
        endpoint_url = "airlab-share-01.andrew.cmu.edu:9000"
        # public key (for donloading): 
        access_key = "4e54CkGDFg2RmPjaQYmW"
        secret_key = "mKdGwketlYUcXQwcPxuzinSxJazoyMpAip47zYdl"

        self.client = Minio(endpoint_url, access_key=access_key, secret_key=secret_key, secure=True)
        self.bucket_name = bucket_name

    def download(self, filelist, destination_path):
        target_filelist = []

        for source_file_name in filelist:
            target_file_name = join(destination_path, source_file_name.replace('/', '_'))
            target_filelist.append(target_file_name)
            print('--')
            if isfile(target_file_name):
                print_error('Error: Target file {} already exists..'.format(target_file_name))
                return False, None

            print(f"  Downloading {source_file_name} from {self.bucket_name}...")
            self.client.fget_object(self.bucket_name, source_file_name, target_file_name)
            print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")

        return True, target_filelist

class CloudFlareDownloader(object):
    def __init__(self, bucket_name = "tartanair-v1") -> None:
        import boto3
        access_key = "f1ae9efebbc6a9a7cebbd949ba3a12de"
        secret_key = "0a21fe771089d82e048ed0a1dd6067cb29a5666bf4fe95f7be9ba6f72482ec8b"
        endpoint_url = "https://0a585e9484af268a716f8e6d3be53bbc.r2.cloudflarestorage.com"

        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3', aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key,
                      endpoint_url=endpoint_url)

    def download(self, filelist, destination_path):
        """
        Downloads a file from Cloudflare R2 storage using S3 API.

        Args:
        - filelist (list): List of names of the files in the bucket you want to download
        - destination_path (str): Path to save the downloaded file locally
        - bucket_name (str): The name of the Cloudflare R2 bucket

        Returns:
        - str: A message indicating success or failure.
        """

        from botocore.exceptions import NoCredentialsError
        target_filelist = []
        for source_file_name in filelist:
            target_file_name = join(destination_path, source_file_name.replace('/', '_'))
            target_filelist.append(target_file_name)
            print('--')
            if isfile(target_file_name):
                print_error('Error: Target file {} already exists..'.format(target_file_name))
                return False, None
            try:
                print(f"  Downloading {source_file_name} from {self.bucket_name}...")
                source_file_name = join('tartanair', source_file_name) # hard code that the cloudflare has a specific prefix folder
                self.s3.download_file(self.bucket_name, source_file_name, target_file_name)
                print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")
            except FileNotFoundError:
                print_error(f"Error: The file {source_file_name} was not found in the bucket {self.bucket_name}.")
                return False, None
            except NoCredentialsError:
                print_error("Error: Credentials not available.")
                return False, None
        return True, target_filelist

    def get_all_s3_objects(self):
        continuation_token = None
        content_list = []
        while True:
            list_kwargs = dict(MaxKeys=1000, Bucket = self.bucket_name)
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token
            response = self.s3.list_objects_v2(**list_kwargs)
            content_list.extend(response.get('Contents', []))
            if not response.get('IsTruncated'):  # At the end of the list?
                break
            continuation_token = response.get('NextContinuationToken')
        return content_list

def get_size(filesizelist, filelist):
    keys_sizes = {rrr[0]: float(rrr[1]) for rrr in filesizelist}
    total_size = 0.
    for ff in filelist:
        total_size += keys_sizes[ff]
    return total_size

def unzip_files(zipfilelist, target_dir):
    print_warn('Note unzipping will overwrite existing files ...')
    for zipfile in zipfilelist:
        if not isfile(zipfile) or (not zipfile.endswith('.zip')):
            print_error("The zip file is missing {}".format(zipfile))
            return False
        print('  Unzipping {} ...'.format(zipfile))
        cmd = 'unzip -q -o ' + zipfile + ' -d ' + target_dir
        system(cmd)
    print_highlight("Unzipping Completed! ")

if __name__ == '__main__':
    args = get_args()

    if args.cloudflare:
        downloader = CloudFlareDownloader()
    else:
        downloader = AirLabDownloader()

    # output directory
    outdir = args.output_dir
    if not isdir(outdir):
        print('Output dir {} does not exists!'.format(outdir))
        exit()

    # difficulty level
    levellist = ['Easy', 'Hard']
    if args.only_easy:
        levellist = ['Easy']
    if args.only_hard:
        levellist = ['Hard']
    if args.only_easy and args.only_hard:
        print('--only-eazy and --only-hard tags can not be set at the same time!')
        exit()


    # filetype
    typelist = []
    if args.rgb:
        typelist.append('image')
    if args.depth:
        typelist.append('depth')
    if args.seg:
        typelist.append('seg')
    if args.flow:
        typelist.append('flow')
    if len(typelist)==0:
        print('Specify the type of data you want to download by --rgb/depth/seg/flow')
        exit()

    # camera 
    cameralist = ['left', 'right', 'flow', 'mask']
    if args.only_left:
        cameralist.remove('right')
    if args.only_right:
        cameralist.remove('left')
    if args.only_flow:
        cameralist.remove('mask')
    if args.only_mask:
        cameralist.remove('flow')
    if args.only_left and args.only_right:
        print('--only-left and --only-right tags can not be set at the same time!')
        exit()
    if args.only_flow and args.only_mask:
        print('--only-flow and --only-mask tags can not be set at the same time!')
        exit()

    # read all the zip file urls
    with open('download_training_zipfiles.txt') as f:
        lines = f.readlines()
    zipsizelist = [ll.strip().split() for ll in lines if ll.strip().split()[0].endswith('.zip')]

    downloadlist = []
    for zipfile, _ in zipsizelist:
        zf = zipfile.split('/')
        filename = zf[-1]
        difflevel = zf[-2]

        # image/depth/seg/flow
        filetype = filename.split('_')[0] 
        # left/right/flow/mask
        cameratype = filename.split('.')[0].split('_')[-1]
        
        if (difflevel in levellist) and (filetype in typelist) and (cameratype in cameralist):
            downloadlist.append(zipfile) 

    if len(downloadlist)==0:
        print('No file meets the condition!')
        exit()

    print_highlight('{} files are going to be downloaded...'.format(len(downloadlist)))
    for fileurl in downloadlist:
        print ('  -', fileurl)

    all_size = get_size(zipsizelist, downloadlist)
    print_highlight('*** Total Size: {} GB ***'.format(all_size))

    # download_from_cloudflare_r2(s3, downloadlist, outdir, bucket_name)
    res, downloadfilelist = downloader.download(downloadlist, outdir)

    if args.unzip:
        unzip_files(downloadfilelist, outdir)
    # for fileurl in downloadlist:
    #     zf = fileurl.split('/')
    #     filename = zf[-1]
    #     difflevel = zf[-2]
    #     envname = zf[-3]

    #     envfolder = outdir + '/' + envname
    #     if not isdir(envfolder):
    #         mkdir(envfolder)
    #         print('Created a new env folder {}..'.format(envfolder))
    #     # else: 
    #     #     print('Env folder {} already exists..'.format(envfolder))

    #     levelfolder = envfolder + '/' + difflevel
    #     if not isdir(levelfolder):
    #         mkdir(levelfolder)
    #         print('  Created a new level folder {}..'.format(levelfolder))
    #     # else: 
    #     #     print('Level folder {} already exists..'.format(levelfolder))

    #     targetfile = levelfolder + '/' + filename
    #     if isfile(targetfile):
    #         print('Target file {} already exists..'.format(targetfile))
    #         exit()

    #     # if args.azcopy:
    #     #     cmd = 'azcopy copy ' + fileurl + ' ' + targetfile 
    #     # else:
    #     cmd = 'wget -r -O ' + targetfile + ' ' + fileurl
    #     ret = system(cmd)

    #     if ret == 2: # ctrl-c
    #         break

