console.log(days);
console.log(stepcount);

const data = []

for(var i = 0; i < days.length; i++) {
    
    data.push({'name': days[i], 'value': stepcount[i], color: '#add7a8'});
    
}

console.log(data);

const width = 300;
const height = 270;
const margin = {top: 40, left: 40, bottom: 40, right: 5};

const x = d3.scaleBand()
  .domain(data.map(d => d.name))
  .range([margin.left, width - margin.right])
  .padding(0.4);

const y = d3.scaleLinear()
  .domain([0, 13000]).nice()
  .range([height - margin.bottom, margin.top]);

const xAxis = g => g
  .attr('transform', `translate(0, ${height - margin.bottom})`)
  .call(d3.axisBottom(x)
    .tickSizeOuter(0)
    .tickFormat(g => g%2==1 ? g : null))
  .call(g => g.select('.domain').remove())
  .call(g => g.selectAll('line').remove());

const yAxis = g => g
  .attr('transform', `translate(${margin.left}, 0)`)
  .call(d3.axisLeft(y)
    .ticks(6))
  .call(g => g.select('.domain').remove())
  .call(g => g.selectAll('line')
    .attr('x2', width)
    .style('stroke', '#f5f5f5')
    .style('stroke-width', 2))
  
   
const svg = d3.select('#vis').append('svg').style('width', width).style('height', height);


svg.append('g').call(xAxis).style("font-size", "10px");
svg.append('g').call(yAxis).style("font-size", "9px");

svg.append('g')
    .selectAll('rect').data(data).enter().append('rect')
    .attr('x', d => x(d.name))
    .attr('y', d => y(d.value))
    .attr('height', d => y(0) - y(d.value))
    .attr("rx", 5)
    .attr('width', x.bandwidth())
    .attr('fill', d => d.color)
    .attr('data-x', d => d.name)
    .attr('data-y', d => d.value)
    .attr('data-color', d=> d.color);

svg.node();