var draw = SVG('drawing');

var mode = 'b'
// can be 'b' for draw box, 'p' for draw polygon, 'e' for edit polygon
// start with 'b'
var manual = false
var useGGNN = true
var crosslines = null;
// group for the crosshair lines
var cprime = null;
// dummy first point
var firstline = null;
// first edge

// Current bbox and polygon
var box = null; 
var polygon = null; 
var prev_polygons = [];
// We will implement this as a stack
var MAX_STACK_SIZE = 10;

// Set of current visible circle SVG elements
var points = Array();

// Img dimension vars
var w = null;
var h = null;
var viewW = null;
var viewH = null;

// params
var circleSize = 2;
var svgPanZoom = null;

// initialization
var base64im = 'static/img/test.jpg';
var defaultim = 'static/img/test.jpg'; // This is a constant
var globalFolder = '/ais/gobi6/amlan/poly_rnn_demo_images/' 
// Put your folder here, it should contain all the images that are present in static/img/
var globalPath = globalFolder + 'test.jpg';
var backend = $('#backend').val();
var imageUploaded = false;

// Set of all polygons, boxes and labels
var polygons = Array();
var boxes = Array();
var labels = Array();
var downloadData = null;

// Check variable
var backend_alerted = false;

basicListeners();
newImage(false);

function shortCutListeners(e){
    if (e.keyCode == 78) {
        $('#n').click();
    }
    else if (e.keyCode == 46) {
        $('#d').click();
    }
    else if (e.keyCode == 65) {
        $('#a').click();
    }
    else if (e.keyCode == 68) {
        $('#d').click();
    }
    else if (e.keyCode == 70) {
        $('#f').click();
    }
    else if (e.keyCode == 71) {
        $('#g').click();
    }
    else if (e.keyCode == 83) {
        $('#s').click();
    }
    else if (e.keyCode == 85) {
        $('#u').click();
    }
}

function startShortCutListeners(){
    document.addEventListener('keydown', shortCutListeners)
}

function stopShortCutListeners(){
    document.removeEventListener('keydown', shortCutListeners)
}

function basicListeners(){
    startShortCutListeners();

    document.addEventListener('click', function(e){
        // Remove pesky points 
        // BIG HACK
        // True solution would be to go into svg.draw.js
        // and remove their drawCircles() function and 
        // recompile the minified file
        elements = draw.select('circle')
        if (elements.members.length > 0){
            elements.members.forEach(function(item, index){
                idx = item.attr('idx');
                if (idx == undefined){
                    item.remove();
                }
            });
        }    
    });

    $('#input').change(function() {
        var file    = document.querySelector('input[type=file]').files[0];
        var reader  = new FileReader();
      
        reader.onloadend = function () {
            base64im = reader.result;
            newImage(false);
        }

        if (file) {
            reader.readAsDataURL(file);
        } else {
            alert('Could not read the file!');
        }
    });

    $("#backend").change(function(){
        backend = $(this).val();
        console.log('Backend changed to: '+backend);
        if (!backend_alerted){
            alert('If you are using your own image, please re-upload it for a full backend change!\nThe models were trained on cityscapes and finetuned on the second dataset (very small for medical/aerial) (if there is a plus).\nPlease check our paper for more details!');
            backend_alerted = true;
        }
    });

    $("#imgselect").change(function(){
        mode = 'z';
        img = $(this).val();
        base64im = 'static/img/'+img;
        globalPath = globalFolder + img;
        if (imageUploaded){
            finish();
        }
        waiting();
        DrawImage();        
    });

    $('#input-label').click(function(){
        // Stop shortcuts so that one can type
        // the label. We will turn it on in changeLabel()
        stopShortCutListeners();
    })
}

function ValidURL(str) {
    var pattern = new RegExp('^(https?:\\/\\/)?'+ // protocol
        '((([a-z\\d]([a-z\\d-]*[a-z\\d])*)\\.?)+[a-z]{2,}|'+ // domain name
        '((\\d{1,3}\\.){3}\\d{1,3}))'+ // OR ip (v4) address
        '(\\:\\d+)?(\\/[-a-z\\d%_.~+]*)*'+ // port and path
        '(\\?[;&a-z\\d%_.~+=-]*)?'+ // query string
        '(\\#[-a-z\\d_]*)?$','i'); // fragment locator
        
        if(!pattern.test(str)) {
            alert("Please enter a valid URL.");
            return false;
        } 
        else {
            return true;
        }
}

function submitURL(){
    var url = $("#url").val();
    if (ValidURL(url)){
        if (imageUploaded){
            finish();
            // alert('Finish with the current image before moving on!');
        }
        // else{
        // CORS Proxy to download imagesls
        url_d = 'https://crossorigin.me/' + url;
        // url_d = 'https://cors-proxy.htmldriven.com/?url=' + url;
        var xhr = new XMLHttpRequest();
        xhr.onload = function() {
            var reader = new FileReader();
            reader.onloadend = function() {
                base64im = reader.result;
                newImage(true);
            }
            reader.readAsDataURL(xhr.response);
        };
        xhr.open('GET', url_d, true);
        xhr.responseType = 'blob';
        xhr.send();
        // }
    }
}

function DrawImage(){
    img = draw.image(base64im).loaded(function(loader){
        waitdone();
        imageUploaded = true;
        w = loader.width;
        h = loader.height;
        
        w_contain = $('#drawing_c').width();
        h_contain = $('#drawing_c').height();       

        if (w_contain*1.0/w < h_contain/h){
            viewW = w_contain;
            viewH = (w_contain*1.0/w)*h
        }
        else{
            viewH = h_contain;
            viewW = (h_contain*1.0/h)*w
        };

        $('#drawing').css('width', viewW);
        $('#drawing').css('height', viewH);

        if (w_contain > viewW){
            $('#drawing').css('margin-left', 0.5*(w_contain-viewW));
        }

        if (h_contain > h){
            $('#help-buttons').css('margin-bottom', h_contain-viewH);
        }

        // Options for the pan zoom library:
        // {
        //     events: {
        //         mouseWheel: true, // enables mouse wheel zooming events
        //         doubleClick: true, // enables double-click to zoom-in events
        //         drag: true, // enables drag and drop to move the SVG events
        //         dragCursor: string "move" // cursor to use while dragging the SVG
        //     },
        //     animationTime: 600, // time in milliseconds to use as default for animations. Set 0 to remove the animation
        //     zoomFactor: 0.1, // how much to zoom-in or zoom-out
        //     maxZoom: 3, //maximum zoom in, must be a number bigger than 1
        //     panFactor: 100, // how much to move the viewBox when calling .panDirection() methods
        //     limits: { // the limits in which the image can be moved. If null or undefined will use the initialViewBox plus 15% in each direction
        //         x: 0
        //         y: 0
        //         x2: viewW
        //         y2: viewH
        //     }
        // }

        draw.viewbox(0,0,w,h);        

        if (svgPanZoom == null){
            svgPanZoom = $('#SvgjsSvg1001').svgPanZoom();
        }

        svgPanZoom.events.doubleClick = false;
        svgPanZoom.events.drag = false;
        svgPanZoom.events.mouseWheel = false;
        svgPanZoom.animationTime = 100;
        svgPanZoom.maxZoom = 5;
        svgPanZoom.panFactor = 600;
        svgPanZoom.limits.x = 0;
        svgPanZoom.limits.y = 0;
        svgPanZoom.limits.x2 = w;
        svgPanZoom.limits.y2 = h;
        svgPanZoom.zoomFactor = 0.05;
        draw.viewbox(0,0,w,h);        

        newObject();            
    });
}

function newImage(isurl){
    waiting();
    if (base64im != defaultim){
        if (!isurl){
            // New Image
            // Upload to server
            metadata_dict = {};
            metadata_dict['image'] = base64im;
            $.ajax({
                method: 'POST',
                url: 'http://' + backend + '/upload_v2',
                contentType: 'application/json',
                data: JSON.stringify(metadata_dict),
                success: function (response){
                    if (response['path'] != ''){
                        globalPath = response['path'];
                        DrawImage();
                    }
                    else{
                        base64im = defaultim;
                        alert ('Error in uploading image! Please try another image.');                
                        waitdone();
                    }
                },
                error: function (reponse) {
                    base64im = defaultim;
                    alert ('Error in uploading image! Please try another image.');
                    waitdone();
                }
            });
        }
        else{
            metadata_dict = {};
            metadata_dict['url'] =  $("#url").val();
            $.ajax({
                method: 'POST',
                url: 'http://' + backend + '/upload_v3',
                contentType: 'application/json',
                data: JSON.stringify(metadata_dict),
                success: function (response){
                    if (response['path'] != ''){
                        globalPath = response['path'];
                        DrawImage();
                    }
                    else{
                        base64im = defaultim;
                        alert ('Error in uploading image! Please try another image.');                
                        waitdone();
                    };
                },
                error: function (reponse) {
                    base64im = defaultim;
                    alert ('Error in uploading image! Please try another image.');
                    waitdone();
                }
            });
        }
        $("#url").val('')
        
    }
    else{
        $("#url").val('')
        DrawImage();
    }
}

function clickImageInput(){
    if (imageUploaded){
        // alert('Finish with the current image before moving on!');
        finish();
    }
    // else{
    $('#input').click();
    // }
}

function modeToggle(){
    if (mode != 'p'){
        manual = !manual;
        
        if (mode == 'e'){
            if (manual){
            }

            else{
            }
            // Switched in editing phase
        }

        if (mode == 'b'){
            // Switched in bbox drawing phase
        }
    }
    else {
        alert('Cannot go to AI-assisted mode while manually drawing polygons!')
        $('#a').prop('checked',false);
    }
}

function ggnnmodeToggle(){
    useGGNN = !useGGNN;
}

function discard(){
    finishObject(true);
}

function stackPrevPolygon(){
    if (prev_polygons.length > MAX_STACK_SIZE){
        prev_polygons.shift();
        // Remove 1st element
    }

    var curr = [];
    currentArray = polygon.array().value

    for (var i = 0; i < currentArray.length; i++)
        curr[i] = currentArray[i].slice();
        // Deep copy

    prev_polygons.push(curr)
    // Make a copy of current polygon array
    // and push to stack
}

function newObject(){
    finishObject(false);
    box = draw.rect().fill('none').stroke({ width: 1, color: 'red' });
    crosslines = draw.group();
    mode = 'b';
    drawer(mode);
}

function changeView(box){
    x = box.x();
    y = box.y();
    width = box.width();
    height = box.height();

    // add 30% expansion
    x = x-(width*0.15);
    y = y-(height*0.15);
    width = width + width*0.3;
    height = height + height*0.3;

    draw.viewbox(x,y,width,height);

    // set adaptive circle size
    // different formulate in comments 

    if (width > height){
        circleSize = Math.max(Math.round(width/30),2);
    }
    else{
        circleSize = Math.max(Math.round(height/30),2);
    }
    // if (w*1.0/width < h*1.0/height){
    //     circleSize = Math.max(Math.round(12*width/w),2);
    // }
    // else{
    //     circleSize = Math.max(Math.round(12*height/h),2);
    // };

    // circleSize = Math.max(Math.round(12*(Math.sqrt(width*height*1.0/(viewH*viewW)))),2);
}

function changeLabel(){
    label = $('#input-label')[0].value;
    if (polygons.length > polygon.attr('idx')){
        // overwrite old label
        idx = polygon.attr('idx');
        labels[idx] = label;
    }
    else{
        // add new label
        labels.push(label);
    }

    var obj=$("#labels").find("option[value='"+label+"']")
    if(obj !=null && obj.length>0){
        // already exists in data list
    }
    else{
        var option = document.createElement('option');
        option.value = label;
        $('#labels').append(option);
    };
    startShortCutListeners();
}

function finishObject(discard){
    draw.off('click');
    waitdone();
    // Remove label box
    $("#u").css("display", "none");
    $("#label").css("display", "none");
    $("#submit-label").css("display", "none");
    $('#input-label')[0].value = "";
    
    if (mode == 'p') discard=true;

    if (mode != 'z'){
        svgPanZoom.setViewBox(0,0,w,h);
    }

    // Store old polygon and make a new one    
    if (polygon){
        polygon.draw('drawstop');
        polygon.attr({'fill-opacity': 0.5});

        if (points.length > 0){
            if (!discard){
                if (polygons.length > polygon.attr('idx')){
                    // overwrite old polygon and box
                    idx = polygon.attr('idx');
                    polygons[idx] = polygon;
                    boxes[idx] = box;
                    box.attr({'stroke-width': 0})
                }
                else{
                    // add new polygon
                    polygons.push(polygon);
                    boxes.push(box);
                    box.attr({'stroke-width': 0})
                }

                polygon.attr('stroke-width', 0);
                polygon.on('mouseenter', function(e){
                    if (mode == 'z'){
                        this.attr({'fill-opacity': 0.8});
                    }
                });
                
                polygon.on('mouseleave', function(e){
                    if (mode == 'z'){
                        this.attr({'fill-opacity': 0.4});
                    }
                })

                polygon.on('click', function(e){
                    if (mode == 'z'){
                        polygon = this;
                        polygon.attr({'stroke-width': 1})                    
                        idx = polygon.attr('idx');
                        $('#input-label')[0].value = labels[idx];
                        box = boxes[idx];
                        box.attr({'stroke-width': 1});
                        changeView(box);
                        drawVertices(this);
                        this.attr({'fill-opacity': 0.4});
                        mode = 'e';
                        drawer(mode);
                    }
                })

                firstline.remove();        
                firstline = null;                    
            }

            else{
                if (polygons.length > polygon.attr('idx')){
                    // Update idxs
                    idx = polygon.attr('idx');
                    for (i=idx+1; i<polygons.length; i++){
                        polygons[i].attr('idx', i-1);
                    }
                    polygons.splice(idx, 1);
                    boxes.splice(idx, 1);
                    labels.splice(idx, 1);
                }

                polygon.draw('drawstop');
                polygon.off('drawstart');
                polygon.off('drawpoint');
                polygon.off('drawupdate');
                polygon.remove();
                box.remove();
                firstline.remove();        
                firstline = null;                    
            }
        }
        else{
            polygon.draw('drawstop');
            polygon.off('drawstart');
            polygon.off('drawpoint');
            polygon.off('drawupdate');
            polygon.remove();
            box.remove();
        }
        polygon = null;
        prev_polygons = [];
    }

    draw.off('mousemove');
    draw.off('mouseleave');

    // Destroy points
    elements = draw.select('circle')
    if (elements.members.length > 0){
        elements.members.forEach(function(item, index){
            item.remove();
        });
    }
    points = Array();
    if (crosslines) crosslines.remove();
    crosslines = null;

    // box.attr('stroke-dasharray', "10,10")

    mode = 'z';
    if (manual) $('#a').click();

    if (imageUploaded){
        svgPanZoom.events.drag = true;
        svgPanZoom.events.mouseWheel = true;
    }

    if (polygons.length > 0){
        // Allow download
        $('#down').css('display', 'inline');
        downloadData = {};
        downPolys = Array();
        polygons.forEach(function(item, index){
            p = item.array().value;
            downPolys.push(p);
        });

        downBoxs = Array();
        boxes.forEach(function(item, index){
            p = [item.x(), item.y(), item.width(), item.height()];
            downBoxs.push(p);
        });

        downloadData['poly'] = downPolys;
        downloadData['bbox'] = downBoxs;
        downloadData['labels'] = labels;
        downloadData = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(downloadData));
        $('#downlink').attr("href",downloadData);
        $('#downlink').attr("download","annotation.json");
    }
    else{
        $('#down').css('display', 'none');
        $('#downlink').attr("href","");
        downloadData = null;
    };

}

function finish(){
    imageUploaded = false;
    // Do whatever saving needs to be done here

    // Destroy
    finishObject(true);    
    draw.clear();
    crosslines = null;
    points = Array();
    polygons = Array();
    boxes = Array();
    $('#drawing_c').css('height', '60%');
    $('#drawing').css('width', '100%');
    $('#drawing').css('height', '100%');
    $('#drawing').css('margin-left', 0);
}

function newPoint(cx, cy, color) {
    var circle = draw.circle(circleSize).fill(color).addClass('polyPoint');
    circle.attr({ 'idx': points.length });
    circle.attr({ 'cx': cx, 'cy': cy });
    points.push(circle);

    return circle
}

function drawVertices(polygon){
    // First destroy old vertices
    if (points.length > 0){
        points.forEach(function(item, index){
            item.remove();
        });
    };
    points = Array();
    arr = polygon.array().value;
    arr.forEach(function(item, index){
        x = item[0];
        y = item[1];
        idx = points.length;
        if (idx == 0) color = '#00ff00';
        else color = '#0066ff';
        newPoint(x,y,color);
    });

    // First edge
    x0 = points[0].attr("cx")
    y0 = points[0].attr("cy")
    x1 = points[1].attr("cx")
    y1 = points[1].attr("cy")
    if (firstline == null){
        firstline = draw.line(x0, y0, x1, y1).stroke({width: 1, color: '#00ff00'})
    }
    else{
        firstline.plot(x0, y0, x1, y1);
    };
};


function beginPolygon(e) {
    // Everything to do when a polygon drawing begins
    // in manual mode
   
    document.addEventListener('keydown', function (e) {
        if (e.keyCode == 13 || e.keyCode == 27) {
            polygon.draw('done');
            polygon.off('drawstart');
            polygon.off('drawpoint');
            polygon.off('drawupdate')
        }
    });

    x = Math.round(e.detail.p['x']);
    y = Math.round(e.detail.p['y']);

    color = '#00ff00';
    
    c = newPoint(x, y, color);
    
    // Invisible circle for mouseover
    cprime = draw.circle(circleSize*3)
    cprime.attr({ 'cx': x, 'cy': y });
    cprime.attr({'fill-opacity': 0})
    cprime.attr({'idx': -1});

    cprime.on('click', function(e){
        polygon.draw('done');
        polygon.off('drawstart');
    });

    cprime.on('mouseenter', function(e){
        c.attr("r", circleSize);
    });

    cprime.on('mouseleave', function(e){
        c.attr("r", circleSize/2);
    });

    c.on('click', function(e){
        polygon.draw('done');
        polygon.off('drawstart');
    });

    c.on('mouseenter', function(e){
        c.attr("r", circleSize);
    });

    c.on('mouseleave', function(e){
        c.attr("r", circleSize/2);
    });

    // whileDrawingPolygon(e);
}

function whileDrawingPolygon(e) {
    x = Math.round(e.detail.p['x']);
    y = Math.round(e.detail.p['y']);
    var color = '#0066ff';

    prev_p = points[points.length-1];
    prev_x = prev_p.attr('cx');
    prev_y = prev_p.attr('cy');

    newPoint(x, y, color);
}

function endPolygon(e) {
    // removes EventListener
    // set to polygon editing mode
    p0 = points[points.length - 1]
    p1 = points[0]

    mode = 'e';
    points[0].attr("r", circleSize/2);
    points[0].off('click');
    points[0].off('mouseleave');
    points[0].off('mouseenter');
    cprime.off('click');
    cprime.off('mouseenter');
    cprime.off('mouseleave');
    cprime.remove();

    // First edge
    x0 = points[0].attr("cx")
    y0 = points[0].attr("cy")
    x1 = points[1].attr("cx")
    y1 = points[1].attr("cy")
    firstline = draw.line(x0, y0, x1, y1).stroke({width: 1, color: '#00ff00'})

    drawer(mode);   
}

function removeCrossHairs(){
    if (crosslines){
        var lineX = crosslines.get(0);
        var lineY = crosslines.get(1);
        lineX.remove();
        lineY.remove();
    }
}

function undo(){
    prev_poly = prev_polygons.pop();
    if (typeof prev_poly != 'undefined'){
        polygon.plot(prev_poly);
        stopEditHandlers(points);
        drawVertices(polygon);
        drawer('e');
    }
    else{
        alert('Reached end of undo stack!');
    };
}

function boxDone(e) {
    // Remove lines and stop mousemove
    // handler
    removeCrossHairs();
    draw.off('mousemove', crosshairs);
    draw.off('mouseleave', removeCrossHairs);
    draw.off('mousedown');
    draw.off('mouseup');

    if (box.width() * box.height() < 200) {
        alert('Box is too small!');
        box.remove();
        newObject();
        return;
    }

    polygon = draw.polygon().fill('#ff66cc').stroke({ width: 1, color:'#660066'});
    polygon.attr("fill-opacity", 0.5);
    polygon.attr("idx", polygons.length);

    changeView(box);

    // set to polygon drawing mode
    mode = 'p';
    drawer(mode);
}

function globalXY(e) {
    var ev = e || window.event;
    
    cx = ev.clientX;
    cy = ev.clientY;

    return [cx, cy];
}

function editColors() {
    // On drag start, stack polygon
    stackPrevPolygon();
    if (manual == true) return;

    idx = $(this).attr("idx");
    for (i = 0; i <= idx; i++) {
        points[i].fill('red');
    }
}

function resetColors() {
    if (manual == true) return;

    idx = $(this).attr("idx");
    points[0].fill('#00ff00');
    for (i = 1; i <= idx; i++) {
        points[i].fill('#0066ff');
    }
}

function movePoint(e) {
    var c = globalXY(e);
    cpt = draw.point(c[0], c[1]);

    // Update point
    $(this).attr("cx", cpt['x']);
    $(this).attr("cy", cpt['y']);

    // Update polygon
    var arr = polygon.array().value;
    idx = $(this).attr("idx");
    arr[idx][0] = cpt['x'];
    arr[idx][1] = cpt['y'];
    polygon.plot(arr);
    
    // In case the first edge moved
    x0 = points[0].attr("cx")
    y0 = points[0].attr("cy")
    x1 = points[1].attr("cx")
    y1 = points[1].attr("cy")
    firstline.plot(x0, y0, x1, y1)
}

function waiting(){
    $('#overlay').css('visibility', 'visible');
    $('#overlay').css('opacity', '1');
};

function waitdone(){
    $('#overlay').css('opacity', '0');
    $('#overlay').css('visibility', 'hidden');
};

function editRequest() {
    resetColors();

    if (!manual){
        waiting();
    
        j = parseInt($(this).attr('idx'));
        metadata_dict = {};
        slice = polygon.array().value.slice(0,j+1);
        metadata_dict['poly'] = slice;
        metadata_dict['bbox'] = [box.x(), box.y(), box.width(), box.height()];
        metadata_dict['img_path'] = globalPath;

        if (useGGNN){
            postURL = 'http://' + backend + '/api/fix_poly_ggnn';
        }
        else{
            postURL = 'http://' + backend + '/api/fix_poly'
        }

        $.ajax({
            url: postURL,
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(metadata_dict),
            success: function (response) {
                stackPrevPolygon();
                poly_to_draw = response['results'][0][0];
                poly_to_draw = poly_to_draw.slice(j+1,-1);
                poly_to_draw = slice.concat(poly_to_draw);
                polygon.plot(poly_to_draw);
                stopEditHandlers(points);
                drawVertices(polygon);
                waitdone();
                drawer('e');
            }
        });
    }
}

function dragPoints(pts){
    pts.forEach(function (item, index) {
        $(item.node).draggable({
            start: editColors,
            drag: movePoint,
            stop: editRequest
        });
    });
}

function stopEditHandlers(pts){
    pts.forEach(function (item, index){
        $(item).off('dblclick');
        $(item).off('contextmenu');
        $(item.node).draggable("destroy");
    })
}

function dblClickAdd(pts){
    pts.forEach(function (item, index) {
        item.on('dblclick', function(e){
            stackPrevPolygon();
            idx = item.attr('idx');
            // Increment index
            for (i=idx+1; i<points.length; i++){
                old_idx = points[i].attr('idx');
                points[i].attr('idx', old_idx+1);
            }

            new_point = item.clone();
            new_point.attr('idx', idx+1);
            new_point.fill('#0066ff');
            new_point.dmove(2,2);
            points.splice(idx+1,0,new_point);
            
            // Update polygon
            var arr = polygon.array().value;
            arr.splice(idx+1,0,[new_point.attr('cx'), new_point.attr('cy')])
            polygon.plot(arr);

            // Add handlers to new point and edge
            editPolygon([points[idx+1]]);
        });
    });
}

function rightClickDel(pts){
    pts.forEach(function(item, index){
        item.on('contextmenu', function(e){
            stackPrevPolygon();
            // Set prev_polygon
            if (points.length > 3){
                idx = item.attr('idx');
                // Decrement index
                for (i=idx+1; i<points.length; i++){
                    points[i].attr('idx', i-1);
                }

                points.splice(idx,1);

                // In case the first point was removed
                points[0].fill('#00ff00');

                x0 = points[0].attr("cx")
                y0 = points[0].attr("cy")
                x1 = points[1].attr("cx")
                y1 = points[1].attr("cy")
                firstline.plot(x0, y0, x1, y1)

                // Update polygon
                var arr = polygon.array().value;
                arr.splice(idx,1)
                polygon.plot(arr);
                // Remove handlers from this point
                stopEditHandlers([item]);
                item.remove();
            }
            else{
                alert('Too few points to delete!')
            }
            e.preventDefault();            
            // to suppress default right click

        })
    });
}

function editPolygon(pts) {
    // Change special options visibility
    $('#down').css('display', 'none');
    // Allow one step undo
    $("#u").css("display", "inline");
    $("#label").css("display", "inline")
    $("#submit-label").css("display", "inline")
    if (prev_polygons.length == 0){
        stackPrevPolygon();
    }

    // add listeners for add mode and remove mode
    // add edit handlers
    dragPoints(pts);
    dblClickAdd(pts);
    rightClickDel(pts);
}

function crosshairs(e) {
    // Listener to mousemove
    // draws crosshairs
    // used during bbox getting

    var c = globalXY(e);
    cpt = draw.point(c[0], c[1]);
    var lineX = crosslines.get(0);
    var lineY = crosslines.get(1);

    if (!lineX) {
        var lineX = crosslines.line(0, cpt['y'], 5000, cpt['y']);
        lineX.attr('id', 'linex');
        lineX.stroke({width: 2, color: 'red'});
        lineX.attr('stroke-dasharray', "10, 10");
    }

    if (!lineY) {
        var lineY = crosslines.line(cpt['x'], 0, cpt['x'], 5000);
        lineY.attr('id', 'liney');
        lineY.stroke({width: 2, color: 'red'});
        lineY.attr('stroke-dasharray', "10, 10");
    }

    lineX.plot(0, cpt['y'], 5000, cpt['y']);
    lineY.plot(cpt['x'], 0, cpt['x'], 5000);
}

function redraw(poly_to_draw){
}

function drawer(mode) {
    svgPanZoom.events.drag = false;
    svgPanZoom.events.mouseWheel = false;

    if (manual) draw_manual(mode);
    else draw_interactive(mode);
}

function ggnn(){
    if (mode != 'e') {
        alert('Can use smoothing only while editing!');
        return;
    }
    stackPrevPolygon();
    metadata_dict['img_path'] = globalPath;
    metadata_dict['bbox'] = [box.x(), box.y(), box.width(), box.height()];
    metadata_dict['poly'] = polygon.array().value;

    $.ajax({
        url: 'http://' + backend + '/api/ggnn_poly',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(metadata_dict),
        success: function (response) {
            poly_to_draw = response['results'][0][0];
            poly_to_draw = poly_to_draw.slice(0,-1);
            polygon.plot(poly_to_draw);
            stopEditHandlers(points);
            drawVertices(polygon);
            $('#a').click();
            drawer('e');
        }
    });

}

function draw_manual(m) {
    if (m == 'p') {
        // does drawing
        polygon.draw()
        // Add events
        polygon.on('drawstart', beginPolygon);
        polygon.on('drawpoint', whileDrawingPolygon)
        polygon.on('drawstop', endPolygon);
    }

    else if (m == 'b') {
        draw.on('mousemove', crosshairs)
        draw.on('mouseleave', removeCrossHairs)

        box.draw();
        box.on('drawstop', boxDone);
    }

    else if (m == 'e') {
        editPolygon(points);
    }
}

function draw_interactive(m) {
    if (m == 'p') {
        waiting();

        metadata_dict = {};
        // img_path in server i.e  base_dir.concat(curr_img)
        metadata_dict['img_path'] = globalPath;
        // id of image, can be anything random
        metadata_dict['image_id'] = 0;
        // [startX , startY , width of bbox, height of bbox ]
        metadata_dict['bbox'] = [box.x(), box.y(), box.width(), box.height()];

        if (useGGNN){
            postURL = 'http://' + backend + '/api/annotation_and_ggnn';
        }
        else{
            postURL = 'http://' + backend + '/api/annotation'
        }

        $.ajax({
            method: 'POST',
            url: postURL,
            contentType: 'application/json',
            data: JSON.stringify(metadata_dict),
            success: function (response) {
                // shape : length, 2
                poly_to_draw = response['results'][0][0];
                // TODO: draw poly
                poly_to_draw = poly_to_draw.slice(0,-1);
                polygon.plot(poly_to_draw);
                drawVertices(polygon);
                waitdone();
                mode = 'e';
                drawer(mode);
            }
        });
    }

    else if (m == 'b') {
        draw.on('mousemove', crosshairs)
        draw.on('mouseleave', removeCrossHairs)
       
        box.draw() 
        //draw.on('mousedown', function(e){
        //    box.draw(e);
        //}, false);

        //draw.on('mouseup', function(e){
        //    box.draw('stop', e);
        //}, false);
        
        box.on('drawstop', boxDone);
    }

    else if (m == 'e') {
        editPolygon(points);
    }
}
