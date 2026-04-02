# Picture Review Workflow

The T7/pictures folder contains four directories for backing up, reviewing and blogging pictures.

## MEDIA_BACKUP

* Copy of backup data from Raspberry Pi 5 SSD.
* **Note:** Do not rename the MEDIA_BACKUP folder or the "yyyy-mm-dd_backup" subfolder.
* These two folders are used by the Pi 5 to rsync backed up pictures and videos to the T7.
* The Pi 5 `backup_pics` project will automatically copy new pictures in the yyyy-mm-dd_backup" subfolder from the Pi 5 to the T7.

## 1. TO_BE_REVIEWED

* Copy backed up pictures and videos from a MEDIA_BACKUP folder to a new folder here.
* Run the `process_images` python program to **File... Reorg Images**.
  * This will organize pictures and videos by day.
  * Drag and drop those into new trip folders.

### Review Process:

* **Generate scores and detect duplicates one trip at a time:**
  * Run the `image_analysis_lib` on a trip folder.
  * `image-analysis score /Volumes/T7/pictures/TO_BE_REVIEWED/{project}`
  * `image-analysis dedupe /Volumes/T7/pictures/TO_BE_REVIEWED/{project}`
* When **File... New** is run, the score and dedupe results will be merged with image info.
* Complete reviewing pictures.
* After completing review, move trip folder to **2. TO_BE_BLOGGED**.

## 2. TO_BE_BLOGGED

* Run `process_images` to:
  * **Show Best**, select all and right click... **Export** — rename folder to `_export_best`.
  * **Show Good or Best** and right click... **Export** — rename folder to `_export_good_best`.

### Blog Entry Steps:

* **For each blog entry:**
  * **UPLOAD** `_export_good_best` pictures for that one blog entry to Google Photos.
  * **Show Best**, select blog photos and right click... **Generate Blog**.
  * **Create blog in blogger.com:**
    * Under blogger account (Douglas.d.garrett@gmail.com), run garrettblog.com.
    * Select **New Post** in upper right corner.
    * Select **NEW POST** button on upper left.
    * Copy generated blog text to entry.
    * **CHANGE** blog date to date of trip entry.
    * Upload blog pictures from `_export_best` folders (X-Large size).
    * Replace variables `{...}` fields; include link to Google Photos album.

### Map Generation:

* After **ALL** blog entries for a trip are created, generate the map.
* Map reads the blog to add labels and links to map pins.
* In `process_images` **Show Best**, select all photos and right click and select **Map**.
* Double click `.html` file to see file in browser.
* Adjust generated HTML zoom factor.
* **BEFORE copying html to ddgarrett.github.io project, CHANGE:**
  * `<script type="text/javascript" src="https://maps.googleapis.com/maps/api/js?libraries=visualization&key=xxxx`
  * See `my_secrets.py` for which `key=` value to use.
* Run `ddgarrett.github.io` project script `build_index.py`.
* Commit map and `index.html` to GitHub.

## 3. DONE

* When done blogging and generating map, move trip to **DONE** folder.
* In `process_images/backup` folder, create a new folder with date and name of trip.
* Copy the `image_collection.csv` file to that folder and commit to GitHub.

### Media Reconciliation:

* Script located in `process_images`: `reconcile_media.py`.
* Generates CSVs in `process_images/backup` directory to track image locations.
* Run after copying trip to another drive.
* Run up to 3 times: on Pi 5, when T7 is attached, and when backup drive is attached.
* Also run when Pi 5 picture backup drive(s) are mounted.

### Google Sheet Sync:

* Copy `process_images/backup/reconcile_media.csv` contents to Google Sheet.
* Use the [**reconcile media**](https://docs.google.com/spreadsheets/d/1RJ7jJOsUNIkhQ_gp5lWvOSsegiBQSywsSy0rPpJR74g/edit?usp=sharing) Google sheet via Mac Numbers app:
  * Open CSV in Numbers and **Copy** all data.
  * Open [reconcile media](https://docs.google.com/spreadsheets/d/1RJ7jJOsUNIkhQ_gp5lWvOSsegiBQSywsSy0rPpJR74g/edit?usp=sharing)  Google sheet and delete existing data.
  * **Paste Special - Data Only** into sheet.
  