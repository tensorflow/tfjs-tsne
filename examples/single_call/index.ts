/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import * as tf_tsne from '../../src/index'
import * as d3 from 'd3';

computeEmbedding();

async function computeEmbedding(){
  let numDimensions = 10;
  let numPoints = 10000;

  //const data = tfc.randomUniform([numPoints, numDimensions]);
  // const data = tfc.tidy(() => {
  //                         return tfc.linspace(0,100,numPoints*numDimensions)
  //                                   .reshape([numPoints, numDimensions]);
  const data = tfc.tidy(() => {
                          return tfc.linspace(0,1,numPoints*numDimensions)
                                    .reshape([numPoints, numDimensions])
                                    .add(tfc.randomUniform([numPoints, numDimensions]));
  });

  data.print();

  const tsne = tf_tsne.tsne(data,{perplexity:30},true);
  await tsne.compute(1000);
  const coordinates = tsne.coordinates();
  const dataCoordinates = await coordinates.data();
  console.log(dataCoordinates);

  var data = [];
  for(let p = 0; p < numPoints; ++p){
    let x = dataCoordinates[p*2];
    let y = dataCoordinates[p*2+1];
    data.push([x, y]);
  }
  appendTheEmbedding(data);
}


/**
 * Appends a new visualization
 */
function appendTheEmbedding(data) {
  const margin = {top: 20, right: 15, bottom: 60, left: 60};
  const width = 800 - margin.left - margin.right;
  const height = 800 - margin.top - margin.bottom;

  let x = d3.scaleLinear()
              .domain([
                d3.min(
                    data,
                    function(d) {
                      return d[0];
                    }),
                d3.max(
                    data,
                    function(d) {
                      return d[0];
                    })
              ])
              .range([0, width]);

  let y = d3.scaleLinear()
              .domain([
                d3.min(
                    data,
                    function(d) {
                      return d[1];
                    }),
                d3.max(
                    data,
                    function(d) {
                      return d[1];
                    })
              ])
              .range([height, 0]);

  let chart = d3.select('body')
                  .append('svg:svg')
                  .attr('width', width + margin.right + margin.left)
                  .attr('height', height + margin.top + margin.bottom)
                  .attr('class', 'chart');

  let main =
      chart.append('g')
          .attr(
              'transform', 'translate(' + margin.left + ',' + margin.top + ')')
          .attr('width', width)
          .attr('height', height)
          .attr('class', 'main');

  let xAxis = d3.axisBottom(x);
  main.append('g')
      .attr('transform', 'translate(0,' + height + ')')
      .attr('class', 'main axis date')
      .call(xAxis);

  let yAxis = d3.axisLeft(y);
  main.append('g')
      .attr('transform', 'translate(0,0)')
      .attr('class', 'main axis date')
      .call(yAxis);

  let g = main.append('svg:g');

  g.selectAll('scatter-dots')
      .data(data)
      .enter()
      .append('svg:circle')
      .attr(
          'cx',
          function(d, i) {
            return x(d[0]);
          })
      .attr(
          'cy',
          function(d) {
            return y(d[1]);
          })
      .attr('stroke-width', 0.25)
      .attr('stroke', '#1f77b4')
      .attr('fill', 'none')
      .attr('r', 5);
}
