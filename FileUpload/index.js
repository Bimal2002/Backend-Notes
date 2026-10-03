// app create
const express= require("express");
const app = express();

// port find karna he
require("dotenv").config();
const PORT= process.env.PORT || 3000;

// middleware add karna h
app.use(express.json());
const fileupload = require("express-fileupload");

app.use(fileupload());

// db se connect karna he
const db = require("./config/database");
db.connect();


// cloud se connnect karna he
const cloudinary= require("./config/cloudinary");
cloudinary.cloudinaryConnect();

//api route mount karna h
const Upload= require("./routes/FileUpload");
app.use("api/v1/upload",Upload);

// active server
app.listen(PORT,()=>{
    console.log(`App started at Port no ${PORT}`);
})

// default route
