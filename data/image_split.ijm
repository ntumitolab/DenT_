dir1 = getDirectory("Choose a Directory");
dir2 = getDirectory("Choose a Directory");
dir3 = getDirectory("Choose a Directory");
inputdir = getDirectory("Choose a Directory");
filelist = getFileList(inputdir);

for (i=0;i<filelist.length;i++){
	file = filelist[i];
	open(inputdir+file);

	//ids=newArray(nImages);
	selectImage(1);
	run("Stack to Images");
	selectImage(1);
	run("Stack to Images");
	selectImage(1);
	run("Stack to Images");

	for (j=0;j<32;j++) {
        selectImage(j+1);
        title = getTitle;
        print(title);
        //ids[i]=getImageID;

        saveAs("tiff", dir1+title);
	}
	for (j=32;j<64;j++) {
        selectImage(j+1);
        title = getTitle;
        print(title);
        //ids[i]=getImageID;

        saveAs("tiff", dir2+title);
	} 
	for (j=64;j<nImages;j++) {
        selectImage(j+1);
        title = getTitle;
        print(title);
        //ids[i]=getImageID;

        saveAs("tiff", dir3+title);
	} 
	run("Close All");
}

