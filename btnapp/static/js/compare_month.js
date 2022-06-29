// 쿼리 결과 값 저장 변수
dataset = [];
dataset1 = [];
var day = ['월', '화', '수', '목', '금', '토', '일']

for(var i=0; i < day.length; i++) {
    if(i >= stepcount_1.length) {
        dataset.push({'name': day[i], 'value': null});
    } else {
        dataset.push({'name': day[i], 'value': parseInt(stepcount_1[i])});
    }  
}

for(var i=0; i < day.length; i++) {
    if(i >= stepcount_2.length) {
        dataset1.push({'name': day[i], 'value': null});
    } else {
        dataset1.push({'name': day[i], 'value': parseInt(stepcount_2[i])});
    } 
}

// set the dimensions and margins of the graph
var width = 300;
var height = 270;
var margin = {top: 40, left: 40, bottom: 40, right: 5};

// append the svg
var svg = d3
  .select("#vis")
  .append("svg")
  .attr("width", width)
  .attr("height", height)
  .append("g")
  .attr("transform", `translate(${margin.left},${margin.top})`);

// group the data
const sumstat = d3.group(data, d => d.name);