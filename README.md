## Bullish Bearish prediction of cryptocurrencies
This repository contains a deep learning model using different architectures to predict the sentimen of the bitcoin prices based on the public perception (posts) and the current and past price.

### Server configuration to run restAPI
    #login to docker image: 
        docker exec -it bit bash
    #then restart apache server:
        service apache2 restart

added to the ~/.bashrc file:

        # This to make sure that apache is running and copying the configuration to run the services.
        ps cax | grep apache > /dev/null
        if [ $? -eq 0 ]; then
        echo "Process is running."
        else
        echo "Process is not running."
        cp /var/www/html/bitpred/config/000-default.conf /etc/apache2/sites-enabled/000-default.conf
        service apache2 start
        fi