<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>View GLTF Model</title>
    <style>
        body { margin: 0; display: flex; }
        canvas { display: block; flex-grow: 1; }
        #nodeList { padding: 20px; background: #f0f0f0; width: 300px; height: 100vh; overflow-y: auto; }
        .node { margin-left: 20px; cursor: pointer; padding: 5px; border-radius: 5px; }
        .node:hover { background-color: #d0d0d0; }
        .node.highlighted { background-color: yellow; }
        #meshInfo {
    padding: 20px;
    background: #f0f0f0;
    width: 300px;
    height: 100vh;
    overflow-y: auto;
    position: fixed;
    top: 0;
    right: 0;
    border-left: 1px solid #ccc; /* Add a border to separate from the main content */
    box-shadow: -2px 0 5px rgba(0, 0, 0, 0.1); /* Add a subtle shadow for depth */
    z-index: 999; /* Ensure it's above other content */
}

    </style>
</head>
<body>
    <div id="nodeList"></div>
    <div id="inputForm">
        <label for="startInput">Start Mesh:</label>
        <select id="startInput"></select>
        <label for="endInput">End Mesh:</label>
        <select id="endInput"></select>
        <button id="drawLineButton">Draw Line</button>
    </div>
    <div id="meshInfo"></div> <!-- Container to display mesh info -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Set up the scene, camera, and renderer
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, (window.innerWidth - 300) / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth - 300, window.innerHeight); // Adjusted to leave space for the node list
        document.body.appendChild(renderer.domElement);

        // Add OrbitControls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1, 0);
        controls.update();

        // Set up lighting
        const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
        directionalLight.position.set(1, 1, 1).normalize();
        scene.add(directionalLight);

        const ambientLight = new THREE.AmbientLight(0xffffff); // Strong white ambient light
        scene.add(ambientLight);

        // Load the GLTF model using the asset URL passed from the Blade template
        const assetUrl = "{{ asset('Landing/mymodel.gltf') }}"; // Blade syntax to generate asset URL
        const loader = new THREE.GLTFLoader();
        loader.load(assetUrl, function(gltf) {
            const model = gltf.scene;
            scene.add(model);

            let meshes = [];

            // Function to traverse the meshes and populate the select options
            function traverseMeshes(mesh, selectElement) {
                if (mesh.isMesh) {
                    meshes.push(mesh);
                    const option = document.createElement('option');
                    option.text = mesh.name || '(no name)';
                    selectElement.add(option);
                }
                mesh.children.forEach(child => traverseMeshes(child, selectElement));
            }

            // Get the node list container and populate it
            const nodeListContainer = document.getElementById('nodeList');
            model.traverse(child => {
                if (child.isMesh) {
                    traverseMeshes(child, document.getElementById('startInput'));
                    traverseMeshes(child, document.getElementById('endInput'));
                }
            });

          
            // Display mesh info
            const meshInfoContainer = document.getElementById('meshInfo');
            const meshData = [];

        // Iterate through meshes to populate the meshData array and display mesh info
        meshes.forEach(mesh => {
            const meshName = mesh.name || '(no name)';
            const meshInfo = {
                name: meshName,
                vertices: []
            };

            if (mesh.geometry && mesh.geometry.attributes.position) {
                const positionAttribute = mesh.geometry.attributes.position;
                const vertexCount = positionAttribute.count;
                const positions = positionAttribute.array;

                for (let i = 0; i < vertexCount; i += 3) {
                    const x = positions[i];
                    const y = positions[i + 1];
                    const z = positions[i + 2];
                    meshInfo.vertices.push({ x, y, z });
                }
            } else {
                console.error(`Geometry or vertices undefined for mesh '${meshName}'`);
            }

            // Push mesh info into the meshData array
            meshData.push(meshInfo);

            // Display mesh name and vertices
            const meshNameElement = document.createElement('div');
            meshNameElement.textContent = `Mesh Name: ${meshName}`;
            meshInfoContainer.appendChild(meshNameElement);

            const meshPointsList = document.createElement('ul');
            meshInfo.vertices.forEach((vertex, index) => {
                const listItem = document.createElement('li');
                listItem.textContent = `Vertex ${index + 1}: (X: ${vertex.x.toFixed(2)}, Y: ${vertex.y.toFixed(2)}, Z: ${vertex.z.toFixed(2)})`;
                meshPointsList.appendChild(listItem);
            });
            meshInfoContainer.appendChild(meshPointsList);
        });
        function drawLine(start, end, color = 0xff0000) {
            console.log(end)
            if (!(start instanceof THREE.Vector3)) {
                start = new THREE.Vector3(start.x, start.y, start.z);
            }
            if (!(end instanceof THREE.Vector3)) {
                end = new THREE.Vector3(end.x, end.y, end.z);
            }

            const material = new THREE.LineBasicMaterial({ color: color });
            const points = [start.clone(), end.clone()]; // Cloning the points to avoid modifying the original vectors
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, material);
            scene.add(line);
        }
        // Handle drawing line when button is clicked
        document.getElementById('drawLineButton').addEventListener('click', function() {
            const startMeshName = document.getElementById('startInput').value;
            const endMeshName = document.getElementById('endInput').value;

            // Find start and end meshes by name from the meshData array
            const startMeshData = meshData.find(mesh => mesh.name === startMeshName);
            const endMeshData = meshData.find(mesh => mesh.name === endMeshName);

            if (startMeshData && endMeshData) {
                // Log start and end mesh names and positions
                // console.log("Start Mesh:", startMeshData.name, "Vertices:", startMeshData.vertices);
                // console.log("End Mesh:", endMeshData.name, "Vertices:", endMeshData.vertices);

                // For demonstration purposes, let's access the vertices of the start and end meshes
                const startVertices = startMeshData.vertices;
                const endVertices = endMeshData.vertices;
                //             const start = new THREE.Vector3(0, 1, 1);
                // const end = new THREE.Vector3(10, 10, 10);
                // drawLine(start, end);
                // Draw lines between corresponding vertices
            console.log( endVertices)
                if (startVertices.length !== endVertices.length) {
                    for (let i = 0; i < startVertices.length; i++) {
                        drawLine(startVertices[i], endVertices[i]);
                        break;
                    }
                } else {
                    console.error('Start and end meshes have different number of vertices.');
                }
            } else {
                console.error('Start or end mesh not found.');
            }
        });


            // Animate the scene
        const animate = function() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();
        }, undefined, function(error) {
            console.error(error);
        });

        // Function to create a line between two points
        function createLine(start, end) {
            console.log("Creating line between:", start, "and", end); // Log start and end positions
            const material = new THREE.LineBasicMaterial({ color: 000 });
            const points = [start, end];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, material);
            scene.add(line);
        }

        // Handle window resize
        window.addEventListener('resize', function() {
            const width = window.innerWidth - 300; // Adjusted to leave space for the node list
            const height = window.innerHeight;
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        });
    </script>
</body>
</html>
