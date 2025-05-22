import glob, pytube as pt, os
import urllib

max_length_saved = 15   # Max video length in minutes (this part broke :( )
max_urls_saved = 10      # 10 was set for Mavceleb v3


with open(glob.glob(os.path.join('**', 'candidate_list'), recursive=True)[0], 'r') as f:

    with open('idList', 'w') as id_list:

        for id, name in enumerate(f.readlines()):

            name = name.strip()
            for language in ['english', 'deutsch']:

                # Get YouTube search results according to name, language, other keywords (list of YouTube objects from pytube)
                res = pt.contrib.search.Search(f'{name} interview {language}').results

                # Save links in files for each language & name
                filepath = os.path.join('identities', name)
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                with open(os.path.join(filepath, f'{language}.txt'), 'w') as out_file:

                    urls_saved = 0
                    for video in res:

                        # Skip videos longer than 15 minutes - NOTE: Fetching the length suddenly throws and error for whatever reason. If you can fix this (or it fixes itself), you can remove the try block
                        try:
                            if video.length() > 60 * max_length_saved:
                                continue
                        except urllib.error.HTTPError as e:
                            #print('oh no')
                            pass

                        # Skip videos where a user needs to be logged in (because of age restrictions)
                        if pt.extract.is_age_restricted(video.watch_url):
                            continue

                        # Write link to file
                        out_file.write(video.watch_url + '\n')
                        urls_saved += 1
                        # Cap the number of links saved
                        if urls_saved >= max_urls_saved:
                            break

            # Write identities into file
            id_list.write(f'{name}\t\tid{id:04}\n')