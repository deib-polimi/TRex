//////////////////////////////////////////////////////////////////
////// Auxiliary functions and constants
//////////////////////////////////////////////////////////////////

function generateUUID() {
    var d = Date.now();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = (d + Math.random()*16)%16 | 0;
        d = Math.floor(d/16);
        return (c=='x' ? r : (r&0x3|0x8)).toString(16);
    });
    return uuid;
};

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

function publish(clientID, evt) {
    var req;
    if(window.XMLHttpRequest) { // Chrome, FireFox, Safari, etc.
	req = new XMLHttpRequest();
    } else if(window.ActiveXObject) { // MSIE
	req = new ActiveXObject("Microsoft.XMLHTTP");
    } else return;
    req.open("PUT", "publish?clientID="+clientID, false);
    req.setRequestHeader("Content-Type", "text/plain");
    req.send("event="+JSON.stringify(evt));
    if(req.status != 200) alert("The request did not succeed!\n\nThe response status was: " +
				req.status + " " + req.statusText + ".");
}

function subscribe(clientID, sub) {
    var req;
    if(window.XMLHttpRequest) { // Chrome, FireFox, Safari, etc.
	req = new XMLHttpRequest();
    } else if(window.ActiveXObject) { // MSIE
	req = new ActiveXObject("Microsoft.XMLHTTP");
    } else return;
    req.open("PUT", "subscribe?clientID="+clientID, false);
    req.setRequestHeader("Content-Type", "text/plain");
    req.send("subscription="+JSON.stringify(sub));
    if(req.status != 200) alert("The request did not succeed!\n\nThe response status was: " +
				req.status + " " + req.statusText + ".");
}

function getevent(clientID) {
    var req;
    var evt;
    if(window.XMLHttpRequest) { // Chrome, FireFox, Safari, etc.
	req = new XMLHttpRequest();
    } else if(window.ActiveXObject) { // MSIE
	req = new ActiveXObject("Microsoft.XMLHTTP");
    } else return;
    req.open("GET", "getevent?clientID="+clientID, false);
    req.setRequestHeader("Content-Type", "text/plain");
    req.send("subscription="+JSON.stringify(evt));
    if(req.status == 200) {
	if(req.responseText.length>0) evt = JSON.parse(req.responseText);
    } else alert("The request did not succeed!\n\nThe response status was: " +
		 req.status + " " + req.statusText + ".");
    return evt;
}

