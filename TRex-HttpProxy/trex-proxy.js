var http = require("http");
var url = require("url");
var fs = require("fs");

var handlers = require("./handlers");

process.on('uncaughtException', function(err) {
    // handle the error safely
    console.log(err);
});


function onRequest(request, response) {
    var pathname = url.parse(request.url).pathname;
    var query = url.parse(request.url).query;
    var postData = '';
    
    request.setEncoding("utf8");
    
    request.on('data', function(chunk) {
	postData += chunk;
    });
    
    request.on('end', function() {
	console.log("About to route a request for " + pathname);
	if (typeof handlers.handle[pathname] === 'function') {
	    handlers.handle[pathname](query, postData, response);
	} else {
	    console.log("No request handler found for " + pathname + " servicing file, instead");
	    fs.readFile('.'+pathname, {encoding: 'utf-8'}, function(err,data){
		if (!err){
		    response.writeHead(200);
		    response.write(data);
		    response.end();
		} else {
		    console.log(err);
		    response.writeHead(404, {"Content-Type": "text/plain"});
		    response.write("404 Not found");
		    response.end();
		}
	    });
	}
    })
}

function start() {
    http.createServer(onRequest).listen(8888);
    console.log("Server has started.");
}

start();

