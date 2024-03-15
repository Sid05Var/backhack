// const express =  require('express');
// const app =  express();
// app.get(
//     '/pythonPage1',
//     (request,response)=>{
//         response.status(200).send('this is pythonPage1')
//     }
// )

// const port =  2000;
// app.listen(
//     port,
//     ()=>{
//         console.log(backend application is running on port ${port})
//     }
// )

/*
const { spawn } = require('child_process');

let obj = {
 name : "shamika",
 rollNo : 39
}

// const childPython = spawn('python', ['--version']);
// const childPython =  spawn('python', ['./PythonData/python3.py']);
// const childPython =  spawn('python', ['./PythonData/python3.py','MIT-WPU']);
// const childPython =  spawn('python', ['./PythonData/python3.py','MIT-WPU',"is in pune" , 34]);
const childPython =  spawn('python', ['./PythonData/python3.py',JSON.stringify(obj)]);


childPython.stdout.on('data', (data) => {
  console.log(stdout =  ${data});
});
childPython.stderr.on('data', (data) => {
  console.error(stderr =  ${data});
});
childPython.on('close', (code) => {
  console.log(child process exited with code  ${code});
});
*/



// Main Code
const express = require("express");
// const child_process = require("child_process");
const { spawn } = require('child_process');
const cors = require('cors');
const app = express();

// Enable CORS with options for specific origins
app.use(cors())//enable cors request
// app.use(cors({
//   origin: ['http://localhost:3002'],
//   credentials: true
// }));

app.get("/run-python", (req, res) => {
//   const { name, rollNo } = req.query;

//   const obj = {
//     name,
//     rollNo,
//   };
let obj = {
 name : "shamika",
 rollNo : 39
}
//   const childPython = child_process.spawn("python", [ "./PythonData/python3.py", JSON.stringify(obj),]);
//   const childPython =  child_process.spawn('python', ['./PythonData/python3.py','MIT-WPU']);
//   const childPython =  child_process.spawn('python', ['./PythonData/python3.py','MIT-WPU',"is in pune" , 34,JSON.stringify(obj)]);
  const childPython =  spawn('python', ['./PythonData/Final3Python.py']);
  // const childPython =  spawn('python', ['./PythonData/python1.py']);
//   const childPython =  spawn('python', ['./PythonData/python3.py',JSON.stringify(obj)]);
  childPython.stdout.on("data", (data) => {
    // console.log(${data});
    // Format the data as JSON and send it back to the client
    // res.json({ data });
    // res.json(${data});
    res.json(`${data}`.replace(/[\r\n]/g, ""));

  });

  childPython.stderr.on("data", (data) => {
    console.error(stderr = `${data}`);
  });

  childPython.on("close", (code) => {
    console.log("child process exited with code ${code}");
  });
});

app.listen(3001, () => {
  console.log("Server listening on port 3001");
})