//////////////////////////////////////////////////////////////////
//// Class Evt
//////////////////////////////////////////////////////////////////

function Evt(evtType, attr) {
    this.evtType = evtType;
    this.timeStamp = 0;
    if (typeof attr === 'undefined') this.attr = {};
    else this.attr = attr;
}

//////////////////////////////////////////////////////////////////
//// Class Sub
//////////////////////////////////////////////////////////////////

function Sub(evtType, constraints) {
    this.evtType = evtType;
    if (typeof constraints === 'undefined') this.constraints = [];
    else this.constraints = constraints;
}

Sub.prototype.addConstraint = function(attrName, operand, value) {
    this.constraints[this.constraints.length] = [attrName, operand, value];
}

Sub.op = {
    EQ: 0,
    LT: 1,
    GT: 2,
    NE: 3,
    IN: 4, 
    LE: 5,
    GE: 6
};

//////////////////////////////////////////////////////////////////
//// Main functions
//////////////////////////////////////////////////////////////////

function getRequest() {
    var req;
    if(window.XMLHttpRequest) { // Chrome, FireFox, Safari, etc.
	req = new XMLHttpRequest();
    } else if(window.ActiveXObject) { // MSIE
	req = new ActiveXObject("Microsoft.XMLHTTP");
    }
    return req;
}

function connect() {
    var req = getRequest();
    var connID;
    req.open("GET", "./connections", false);
    req.send();
    if(req.status == 200 && req.responseText.length>0) {
	connID = JSON.parse(req.responseText);
    } else {
	alert("The request did not succeed!\n\nThe response status was: " +
	      req.status + " " + req.statusText + ".");
    }
    return connID;
}    

function subscribe(connID, sub) {
    var req = getRequest();
    req.open("POST", "./subscriptions/"+connID, false);
    req.setRequestHeader("Content-Type", "application/json");
    req.send(JSON.stringify(sub));
    if(req.status != 200) {
	alert("The request did not succeed!\n\nThe response status was: " + req.status + " " + req.statusText + ".");
    }
    return JSON.parse(req.responseText);    
}

function unsubscribe(connID, sub) {
    var req = getRequest();
    req.open("DELETE", "./subscriptions/"+connID, false);
    req.setRequestHeader("Content-Type", "application/json");
    req.send(JSON.stringify(sub));
    if(req.status != 200) {
	alert("The request did not succeed!\n\nThe response status was: " + req.status + " " + req.statusText + ".");
    }
    return JSON.parse(req.responseText);    
}

function getevent(connID) {
    var req = getRequest();
    var evt;
    req.open("GET", "./events/"+connID, false);
    req.send();
    if(req.status == 200 && req.responseText.length>0) {
	evt = JSON.parse(req.responseText);
    } else if(req.status == 204) {
	return ;
    } else {
	alert("The request did not succeed!\n\nThe response status was: " +
	      req.status + " " + req.statusText + ".");
    }
    return evt;
}

function publish(connID, evt) {
    var req = getRequest();
    req.open("POST", "./events/"+connID, false);
    req.setRequestHeader("Content-Type", "application/json");
    req.send(JSON.stringify(evt));
    if(req.status != 200) {
	alert("The request did not succeed!\n\nThe response status was: " + req.status + " " + req.statusText + ".");
    }
    return JSON.parse(req.responseText);
}
