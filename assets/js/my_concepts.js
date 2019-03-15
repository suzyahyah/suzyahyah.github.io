function generateGraph(nodes){

  for (i=0; i<nodes.length; i++){
    console.log(nodes[i].title);
    
  }

}

function generateEdges(nodes){
  
  var edges = [];
  var categories = {};

  for (i=0; i<nodes.length; i++){
    var node = nodes[i];
    node.size=50;
    for (c=0; c<node.category.length; c++){
      try {
        categories[node.category[c]].push(node.id);
      } catch (err) {
        categories[node.category[c]] = [node.id];
      }
    }
  }


  Object.keys(categories).forEach(function(key) {
    //console.log(categories[key]);
    //
    console.log(categories[key].length);
    var color = '#' + (Math.random().toString(16) + "000000").substring(2, 8)
    const width = 2;

    for (c=0; c<categories[key].length; c++){

      for (j=0; j<categories[key].length; j++){
        if (c===j){
            //pass
        } else{
          
          edges.push({from: categories[key][c], to: categories[key][j], color: {color:color}, title: key, width: width, hoverWidth:width, selectionWidth: width});
        }
        
      }
    }
  });
  return edges;
}


