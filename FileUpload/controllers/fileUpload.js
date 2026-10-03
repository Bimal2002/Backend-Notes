const File = require("../models/File");

// localfileupload -> handler function

exports.localFileUpload = async (req, res) => {
    try {
        // fetch file
        const file = req.files.file;
        console.log("File AAGYI JEE ", file);
        let path = __dirname + "/files/" + Date.now();
        console.log("PATH ->", path)
        file.mv(path, (error) => {
            console.log(err);
        });
        res.json({
            success: true,
            message: 'Local file Uploaded Successfully'
        })
    }
    catch (error) {
        console.log(error)
    }
}