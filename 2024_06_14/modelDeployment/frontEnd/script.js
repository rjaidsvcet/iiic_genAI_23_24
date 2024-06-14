document.getElementById('formData').addEventListener('submit', predictSpam)

function predictSpam (e) {
    e.preventDefault()

    let text = document.getElementById('textarea_1').value

    fetch('http://localhost:5000', {
        method : 'POST',
        headers : {
            'Content-Type' : 'application/json'
        },
        body : JSON.stringify({
            'text' : text
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.response == 'spam') {
            document.getElementById('output').innerHTML = `<div class="alert card red lighten-4 red-text text-darken-4"><div class="card-content">The message is Spam</div></div>`
        }
        else {
            document.getElementById('output').innerHTML = `<div class="alert card green lighten-4 green-text text-darken-4"><div class="card-content">The message is not Spam</div></div>`
        }
    })
}