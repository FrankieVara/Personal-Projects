function updateLabel() {
	var input = document.getElementById('file');
	var label = document.getElementById('file-label');
	var fileName = '';
	if (input.files && input.files.length > 1)
		fileName = `${input.files.length} files selected`;
	else
		fileName = input.files[0].name;

	label.innerHTML = fileName; // Update the entire label content
}